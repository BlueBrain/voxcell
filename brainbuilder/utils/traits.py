'''library to handle traits fields and collections and the logic to assign them

    A "trait" is a set of properties and their values.
    A "traits collection" is a group of traits. It's represented as a pandas
    DataFrame object where each trait is a row and each property is a column.
    For example:
           sclass      mtype
        0  inhibitory  Pyramidal
        1  excitatory  Pyramidal
        2  inhibitory  Martinotti

    A "probability distribution" specifies the probability of each trait in a collection
    being assigned to a given cell.
    A "distributions collection" is a group of distributions. It's represented
    as a pandas DataFrame object where each trait is a row and each distribution
    is a column. For example:
            0     1    2     3
         0  0.25  0.5  0.5   0.0
         1  0.75  0.5  0.25  0.1
         2  0.0   0.0  0.25  0.9

    The distribution number 3 in the above table indicates 10% chances of picking
    the trait 1 (an excitatory Pyramidal cell) and 90% of picking trait 2
    (an inhibitory Martinotti cell).
'''

import numpy as np
import pandas as pd
from collections import namedtuple
from brainbuilder.utils import genbrain as gb


import logging
L = logging.getLogger(__name__)


# TODO decide: should voxel_dimensions also be part of SpatialDistribution?
SpatialDistribution = namedtuple('SpatialDistribution',
                                 'field distributions traits')
# field: volume data where every voxel is an index in the distributions list
#        -1 means unavailable data.
# distributions: a distributions collection, see module docstring
# traits: a traits collection, see module docstring


def assign_from_spatial_distribution(positions, spatial_dist, voxel_dimensions):
    '''for every cell in positions, chooses a property from a spatial distribution

    Args:
        positions: list of positions for soma centers (x, y, z).
        spatial_dist: a spatial distribution of traits
        voxel_dimensions: the size of the voxels in each dimension

    Returns:
        An array with the same length as positions where each value
        is an index into spatial_dist.traits
    '''
    voxel_idx = gb.cell_positions_to_voxel_indices(positions, voxel_dimensions)

    voxel_idx_tuple = tuple(voxel_idx.transpose())
    dist_id_per_position = spatial_dist.field[voxel_idx_tuple]

    unknown_count = np.count_nonzero(dist_id_per_position == -1)
    if unknown_count:
        L.warning('%d total positions in unknown areas', unknown_count)

    valid = np.in1d(dist_id_per_position, spatial_dist.distributions.columns)
    if np.count_nonzero(~valid):
        L.warning('missing distribution for %d positions', np.count_nonzero(~valid))

    chosen_trait_indices = np.ones_like(dist_id_per_position) * -1

    unique_dists = np.unique(dist_id_per_position[valid])
    for dist_id, dist in spatial_dist.distributions[unique_dists].iteritems():
        hit_count = np.count_nonzero(dist_id_per_position == dist_id)

        chosen = np.random.choice(dist.keys(), hit_count, p=dist.values)
        chosen_trait_indices[dist_id_per_position == dist_id] = chosen

    return chosen_trait_indices


# TODO review when we use the terms "probabilities", "distribution", etc and be more consitent
# see module docstring


def normalize_distribution_collection(distribution_collection):
    '''take a collection of probability distributions and normalize them'''
    return distribution_collection / distribution_collection.sum()


def split_distribution_collection(spatial_dist, attributes):
    '''split a distribution in two or more so that each one only references
    traits with the same value for certain attributes.
    Each resulting distribution is renormalised.

    this may be generating distributions that are empty but that should not be a problem

    Note that because for every distribution we are creating a new one,
    the indexes of any associated field are still valid.

    Returns:
        A dictionary where the keys are tuples with the values of the attributes found
        in the traits_collection and the values are the resulting SpatialDistribution objects.
    '''
    grouped_distributions = {}
    for attr_values, traits in spatial_dist.traits.groupby(attributes):

        dists = spatial_dist.distributions.ix[traits.index]

        # remove dists that have become empty
        dists = dists[dists.columns[dists.sum() != 0]]
        dists = normalize_distribution_collection(dists)

        grouped_distributions[attr_values] = SpatialDistribution(spatial_dist.field, dists, traits)

    return grouped_distributions


def reduce_distribution_collection(spatial_dist, attribute):
    '''given a spatial distribution, extract an equivalent one where all of the properties
    of the traits collection have been removed except for a specific one.

    For example:

        Taking the traits collection:
               sclass      mtype
            0  inhibitory  Pyramidal
            1  excitatory  Pyramidal
            2  inhibitory  Martinotti

        And the distributions collection:
               0
            0  0.2
            1  0.4
            2  0.4

        Ignoring all properties except 'sclass' would give the simplified traits collection:
               sclass
            0  inhibitory
            1  excitatory

        And the distributions collection:
               0
            0  0.6
            1  0.4

    Note that because for every distribution we are creating a new one,
    the indexes of any associated field are still valid.

    Returns:
        An SpatialDistribution object
    '''

    distributions = dict((value, spatial_dist.distributions.ix[data.index].sum())
                         for value, data in spatial_dist.traits.groupby(attribute))

    traits = pd.DataFrame(distributions.keys(), columns=[attribute])
    distributions = pd.DataFrame(distributions.values(),
                                 columns=spatial_dist.distributions.columns,
                                 index=traits.index)

    return SpatialDistribution(spatial_dist.field, distributions, traits)
