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


def count_regions_per_points(points, annotation, annotation_transform=None):
    """ Find the regions for given points, count number of points per region. """
    result = defaultdict(int)
    if len(points) == 0:
        return result
    ids = annotation.lookup(points, outer_value=0)
    for id_, count in zip(*np.unique(ids, return_counts=True)):
        if annotation_transform is not None:
            id_ = annotation_transform.get(id_)
        result[id_] += count
    return result


def segment_region_histogram(cells, morphologies, annotation, annotation_transform=None):
    """
        Calculate segment count per region.

        Arguments:
            cells: pandas DataFrame with cell properties
            morphologies: MorphologyLoader to retrieve morphology data
            annotation: VoxelData with region atlas
            annotation_transformation: a dict to group and rename regions

        Returns:
            pandas DataFrame with number of segments per region.
            Multi-indexed by gids and branch types.
            Columns: region ids/names (None for segments with no corresponding region)

            For example:
            gid | branch_type || SLM | SR | SP | SO | NaN |
             42 | axon        ||  11 | 22 | 33 | 44 |   0 |
             42 | dendrite    ||   0 |  0 |  0 |  0 | 100 |

        Comments:
            For example, to sum up values for all SLM regions with ids 12, 13, 14,
            one would pass `annotation_transform={12: 'SLM', 13: 'SLM', 14: 'SLM'}`.
    """
    index = []
    result = []
    for gid, cell in cells.iterrows():
        nrn = morphologies.get(cell['morphology'])
        translation = cell[['x', 'y', 'z']].values.astype(np.float)
        orientation = cell['orientation']
        for branch_type in [nm.APICAL_DENDRITE, nm.AXON, nm.BASAL_DENDRITE]:
            points = nm.get('segment_midpoints', nrn, neurite_type=branch_type)
            if len(points) > 0:
                points = np.array(points)
                points = np.dot(orientation, points.transpose()).transpose()
                points += translation
            values = count_regions_per_points(points, annotation, annotation_transform)
            # pylint: disable=maybe-no-member
            index.append((gid, branch_type.name))
            result.append(values)

    index = pd.MultiIndex.from_tuples(index, names=['gid', 'branch_type'])
    result = pd.DataFrame(result, index=index).fillna(0).astype(int).sort_index()

    return result
