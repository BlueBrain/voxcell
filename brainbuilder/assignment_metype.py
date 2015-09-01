'''algorithm to assign me-types to a group of cells'''
import itertools

import numpy as np

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
        An array of tuples containing the mtype and etype values that correspond to each position.
        For those positions whose me-type could not be determined, nan is used.

    Note:
        Can get the list of METypes from the recipe:
        mtypes = list(set(recipe_sdist.traits['mtype'].values))
        etypes = list(set(recipe_sdist.traits['etype'].values))
    '''
    metypes = np.array(list(itertools.product(mtypes, etypes)), dtype='object')
    choices = np.random.randint(len(metypes), size=len(positions))
    return metypes[choices]


def assign_metype(positions, chosen_sclass, recipe_sdist, voxel_dimensions):
    '''for every cell in positions, assign me-type to each cell based on its synapse class

    Args:
        positions: list of positions for soma centers (x, y, z).
        chosen_sclass: a list of synapse class values that correspond to each position.
        recipe_sdist: SpatialDistribution containing at least the properties: sClass, mtype, etype.
        voxel_dimensions: tuple with the size of the voxels in microns in each axis, (x, y, z)

    Returns:
        An np.array of lists containing the mtype and etype values that correspond to each position.
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

    return recipe_sdist.traits[['mtype', 'etype']].ix[chosen_metype].as_matrix()


def serialize_assign_metype(dst_file, assigned_metypes):
    '''Serialize assigned me types

    Args:
        dst_file(str): fullpath to filename to write
        assigned_metype: list of 2 element tuples (or lists)
    '''
    with open(dst_file, 'w') as fd:
        for metype in assigned_metypes:
            fd.write('%s %s\n' % tuple(metype))


def deserialze_assign_metype(src_file):
    '''De-serialize assigned me types

    Args:
        src_file(str): fullpath to filename to write

    Returns:
        metypes np.array of tuples
    '''
    metypes = []
    with open(src_file, 'r') as fd:
        for line in fd.readlines():
            metypes.append(line.strip().split())

    return np.array(metypes)
