""" Helper functions for validation. """

import os
import numpy as np
import pandas as pd

from collections import defaultdict

import neurom as nm


# TODO: combine with voxcellview.MorphologyLoader and/or move to NeuroM??
class MorphologyLoader(object):
    """ Class to access morphology data by name. """
    def __init__(self, morphology_dir, file_extension='.h5'):
        self.morphology_dir = morphology_dir
        self.file_extension = file_extension

    def _get_filepath(self, name):
        # pylint: disable=missing-docstring
        return os.path.join(self.morphology_dir, name + self.file_extension)

    def get(self, name):
        """ NeuroM data for given morphology. """
        return nm.load_neuron(self._get_filepath(name))


def _affine_transform(points, translation, rotation):
    """ Apply affine transform to `points`.

        First rotate using `rotation` matrix, then translate with `translation` vector.
    """
    result = np.array(points)
    if len(result) > 0:
        result = np.dot(rotation, result.transpose()).transpose()
        result += translation
    return result


def _count_regions_per_points(points, annotation, annotation_transform=None, weights=None):
    """ Find the regions for given points, count number of points per region. """
    result = defaultdict(int)
    if len(points) == 0:
        return result
    ids = annotation.lookup(points, outer_value=0)
    if weights is None:
        grouped = zip(*np.unique(ids, return_counts=True))
    else:
        grouped = zip(ids, weights)
    for id_, w in grouped:
        if annotation_transform is not None:
            id_ = annotation_transform(id_)
        result[id_] += w
    return result


def segment_region_histogram(
    cells, morphologies, annotation, annotation_transform=None, normalize=False, by='count'
):
    """
        Calculate segment count per region.

        Arguments:
            cells: pandas DataFrame with cell properties
            morphologies: MorphologyLoader to retrieve morphology data
            annotation: VoxelData with region atlas
            annotation_transformation: a function to group or rename regions
            normalize: output fractions instead of segment counts
            by: count number of segments (='count'), their length (='length') or volume (='volume')

        Returns:
            pandas DataFrame with number of segments per region.
            Multi-indexed by gids and branch types.
            Columns: region ids/names

            For example:
            gid | branch_type || SLM | SR | SP | SO | NaN |
             42 | axon        ||  11 | 22 | 33 | 44 |   0 |
             42 | dendrite    ||   0 |  0 |  0 |  0 | 100 |

        Comments:
            For example, to sum up values for all SLM regions with ids 12, 13, 14,
            one would pass `annotation_transform={12: 'SLM', 13: 'SLM', 14: 'SLM'}`.
    """
    # pylint: disable=too-many-locals
    BY_ALTERNATIVES = ('count', 'length', 'volume')
    if by not in BY_ALTERNATIVES:
        raise ValueError(
            "Invalid 'by' argument: '{0}', should be one of {1}".format(by, BY_ALTERNATIVES)
        )

    index = []
    result = []
    for gid, cell in cells.iterrows():
        nrn = morphologies.get(cell['morphology'])
        translation = cell[['x', 'y', 'z']].values.astype(np.float)
        orientation = cell['orientation']
        for branch_type in [nm.APICAL_DENDRITE, nm.AXON, nm.BASAL_DENDRITE]:
            points = nm.get('segment_midpoints', nrn, neurite_type=branch_type)
            if by == 'length':
                weights = nm.get('segment_lengths', nrn, neurite_type=branch_type)
            elif by == 'volume':
                weights = nm.get('segment_volumes', nrn, neurite_type=branch_type)
            else:
                weights = None
            points = _affine_transform(points, translation, orientation)
            values = _count_regions_per_points(points, annotation, annotation_transform, weights)
            # pylint: disable=maybe-no-member
            index.append((gid, branch_type.name))
            result.append(values)

    index = pd.MultiIndex.from_tuples(index, names=['gid', 'branch_type'])
    result = pd.DataFrame(result, index=index).fillna(0).sort_index()

    if normalize:
        result = result.div(result.sum(axis=1), axis=0).astype(float)
    elif by == 'count':
        result = result.astype(int)
    else:
        result = result.astype(float)

    return result
