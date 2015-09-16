'''algorithm to assign me-types to a group of cells'''
import itertools

import numpy as np
import pandas as pd

from brainbuilder.utils import traits as tt

import logging
L = logging.getLogger(__name__)


def assign_metype_random(positions, mtypes, etypes):
    '''for every cell in positions, assign me-type to each cell randomly
    Args:
        positions: list of positions for soma centers (x, y, z).
        mtypes: list of all mtypes
        etypes: list of all etypes

    Returns:
        A pandas DataFrame with one row for each position and two columns: mtype and etype.
        For those positions whose me-type could not be determined, nan is used.

    Note:
        Can get the list of METypes from the recipe:
        mtypes = list(set(recipe_sdist.traits['mtype'].values))
        etypes = list(set(recipe_sdist.traits['etype'].values))
    '''
    metypes = np.array(list(itertools.product(mtypes, etypes)), dtype='object')
    choices = np.random.randint(len(metypes), size=len(positions))
    return pd.DataFrame(data=metypes[choices], columns=('mtype', 'etype'))


def assign_metype(positions, chosen_sclass, recipe_sdist, voxel_dimensions):
    '''for every cell in positions, assign me-type to each cell based on its synapse class

    Args:
        positions: list of positions for soma centers (x, y, z).
        chosen_sclass: a list of synapse class values that correspond to each position.
        recipe_sdist: SpatialDistribution containing at least the properties: sClass, mtype, etype.
        voxel_dimensions: tuple with the size of the voxels in microns in each axis, (x, y, z)

    Returns:
        A pandas DataFrame with one row for each position and two columns: mtype and etype.
        For those positions whose me-type could not be determined, nan is used.
    '''
    subsections = tt.split_distribution_collection(recipe_sdist, ('sClass',))

    chosen_metype = np.ones(shape=(len(chosen_sclass)), dtype=np.int) * -1

    for value, subdist in subsections.iteritems():
        mask = np.array(chosen_sclass) == value
        chosen_metype[mask] = tt.assign_from_spatial_distribution(positions[mask],
                                                                  subdist,
                                                                  voxel_dimensions)

    invalid_metype_count = np.count_nonzero(chosen_metype == -1)
    if invalid_metype_count:
        # this may happen because of inconsistencies of the data
        # for example if we assigned excitatory neurons to a neuron that is in
        # a voxel for which only inhibitory metype probabilities are known
        L.warning('%d / %d = %f cells could not get a valid metype assigned',
                  invalid_metype_count, len(chosen_metype),
                  float(invalid_metype_count) / len(chosen_metype))

    df = recipe_sdist.traits[['mtype', 'etype']].ix[chosen_metype]
    return df.reset_index().drop('index', 1)
