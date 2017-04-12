'''algorithm to assign me-types to a group of cells'''
import itertools
import logging

import numpy as np
import pandas as pd


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


def assign_metype(positions, synapse_class, sdist):
    '''for every cell in positions, assign me-type to each cell based on its synapse class

    Args:
        positions: list of positions for soma centers (x, y, z).
        synapse_class: a list of synapse class values that correspond to each position.
        sdist: SpatialDistribution containing at least the properties:
            synapse_class, mtype, etype.

    Returns:
        A pandas DataFrame with one row for each position and two columns: mtype and etype.
        For those positions whose me-type could not be determined, nan is used.
    '''
    chosen = sdist.assign_conditional(positions, synapse_class)
    return sdist.collect_traits(chosen, ['mtype', 'etype'])
