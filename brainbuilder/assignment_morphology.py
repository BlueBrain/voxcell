'''algorithm to assign morphologies to a group of cells'''
from brainbuilder.utils import traits as tt
import numpy as np

import logging
L = logging.getLogger(__name__)


def assign_morphology(positions, chosen_me, sdist):
    '''for every cell in positions, assign a morphology to each cell based on its metype

    Args:
        positions: list of positions for soma centers (x, y, z).
        chosen_me: dataframe with the mtype and etype values that correspond to each position.
        sdist: SpatialDistribution containing at least the properties:
            mtype, etype, morphology.
        voxel_dimensions: tuple with the size of the voxels in microns in each axis.

    Returns:
        A pandas DataFrame with one row for each position and one column: morphology.
        For those positions whose morphology could not be determined, nan is used.
    '''
    subsections = tt.split_distribution_collection(sdist, ('mtype', 'etype'))

    chosen_morphs = np.ones(shape=(len(chosen_me)), dtype=np.int) * -1

    unique_me = chosen_me.drop_duplicates()

    for values_comb in unique_me.values:
        subdist = subsections[tuple(values_comb)]
        mask = np.all(np.array(chosen_me) == values_comb, axis=1)
        chosen_morphs[mask] = tt.assign_from_spatial_distribution(positions[mask], subdist)

    #TODO: docstring says non represented ones are NaN, but here it's looking for -1?
    if np.count_nonzero(chosen_morphs == -1):
        # this may happen because of inconsistencies of the data
        # for example if we assigned the pyramidal mtype to a neuron that is in
        # a voxel for which we only know the distribution of marttinoti morphologies
        L.warning('%d / %d cells could not get a valid morphology assigned',
                  np.count_nonzero(chosen_morphs == -1), len(chosen_morphs))

    df = sdist.traits['morphology'].ix[chosen_morphs]
    return df.reset_index().drop('index', 1)
