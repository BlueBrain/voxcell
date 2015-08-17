'''algorithm to assign morphologies to a group of cells'''
from brainbuilder.utils import traits as tt
import pandas as pd
import numpy as np

import logging
L = logging.getLogger(__name__)


def assign_morphology(positions, chosen_me, spatial_dist, voxel_dimensions):
    '''for every cell in positions, assign a morphology to each cell based on its metype

    Args:
        positions: list of positions for soma centers (x, y, z).
        chosen_me: a list of metype values that correspond to each position.
        spatial_dist: SpatialDistribution containing at least the properties:
            mtype, etype, morphology.
        voxel_dimensions: tuple with the size of the voxels in microns in each axis.

    Returns:
        An array of the morpholgies that correspond to each position.
        For those positions whose morphology could not be determined, nan is used.
    '''
    subsections = tt.split_distribution_collection(spatial_dist, ('mtype', 'etype'))

    chosen_morphs = np.ones(shape=(len(chosen_me)), dtype=np.int) * -1

    unique_me = pd.DataFrame(chosen_me).drop_duplicates()

    for values_comb in unique_me.values:
        subdist = subsections[tuple(values_comb)]
        mask = np.all(np.array(chosen_me) == values_comb, axis=1)
        chosen_morphs[mask] = tt.assign_from_spatial_distribution(positions[mask],
                                                                  subdist,
                                                                  voxel_dimensions)

    if np.count_nonzero(chosen_morphs == -1):
        # this may happen because of inconsistencies of the data
        # for example if we assigned the pyramidal mtype to a neuron that is in
        # a voxel for which we only know the distribution of marttinoti morphologies
        L.warning('%d / %d cells could not get a valid morphology assigned',
                  np.count_nonzero(chosen_morphs == -1), len(chosen_morphs))

    return spatial_dist.traits['morphology'].ix[chosen_morphs].as_matrix()
