'''
    library to handle traits fields and collections and the logic to assign them

    A "trait" is a set of properties and their values. It's represented as a dict.
    For instance:
        {'sclass': 'inhibitory', 'mtype': 'Pyramidal'}

    A "traits collection" is a group of traits. It's represented as a list of dicts.
    For instance:
        [{'sclass': 'inhibitory', 'mtype': 'Pyramidal'},
         {'sclass': 'inhibitory', 'mtype': 'Martinotti'}]

    A "probability distribution" specifies the probability of each trait in a collection
    being assigned to a given cell. It's represented as a dict where the keys
    identify particular traits in a traits collection (they are indices)
    For instance:
        {0: 0.25, 1: 0.75}
    That means 25% chances of picking an inhibitory Pyramidal cell, 75% of picking
    an inhibitory Martinotti cell.

    A "distributions collection" is a group of distributions. For instance:
        [{0: 0.25, 1: 0.75},
         {0: 0.5,  1: 0.5}]
'''

import numpy as np
import h5py
import json
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple

from brainbuilder.utils import genbrain as gb


import logging
L = logging.getLogger(__name__)


SpatialDistribution = namedtuple('SpatialDistribution',
                                 'field distributions traits')
# field: volume data where every voxel is an index in the distributions list
#        -1 means unavailable data.
# distributions: a distributions collection, see module docstring
# traits: a traits collection, see module docstring


def save_traits_collection(filename, all_traits):
    '''save a collection of traits
    all_traits must be a list of dictionaries flat dictionaries, where the keys and values are
    jsonable objects (most probably strings)
    '''
    with open(filename, 'w') as f:
        json.dump(all_traits, f, indent=2)


def load_traits_collection(filename):
    '''load a collection of traits'''
    with open(filename, 'r') as f:
        return json.load(f)


def save_traits_field(filename, voxel_dimensions, field, probabilities):
    '''save a field expressing groups of probability distributions of different traits for each
    one of the voxels.

    Accepts:
        voxel_dimensions: in microns, how big are the voxels, for example: (25, 25, 25)
        field: volume data where veach voxel value is an index in the probabilites object
        probabilities: list where each item is a dictionary that contains traits indices and
            their associated probabilities
    '''
    with h5py.File(filename, 'w') as h5:
        h5.create_dataset(name='field', data=field)

        pg = h5.create_group('probabilities')
        for i, combo in enumerate(probabilities):
            keys, values = zip(*combo.items())
            g = pg.create_group(str(i))
            g.create_dataset('keys', data=np.array(keys))
            g.create_dataset('values', data=np.array(values))

        h5.create_dataset(name='voxel_dimensions', data=voxel_dimensions)


def load_traits_field(filename):
    '''save a field expressing groups of probability distributions of different traits for each
    one of the voxels.
    Returns:
        voxel_dimensions: in microns, how big are the voxels, for example: (25, 25, 25)
        field: volume data where veach voxel value is an index in the probabilites object
        probabilities: list where each item is a dictionary that contains traits and
            their associated probabilities. The traits are expressed as indices into a traits
            collection (see save_traits_collection)
    '''
    with h5py.File(filename, 'r') as h5:
        field = np.array(h5['field'])

        probabilities = []
        pg = h5['probabilities']
        for i in range(len(pg.keys())):
            keys = list(pg[str(i)]['keys'])
            values = list(pg[str(i)]['values'])
            probabilities.append(dict(zip(keys, values)))

        voxel_dimensions = np.array(h5['voxel_dimensions'])

    return voxel_dimensions, field, probabilities


def save_chosen_traits(filename, chosen):
    '''save a collection of chosen traits in h5.
    The collection is just a numpy array.
    The tratis are expressed as indices in a traits collection that is saved independently
    (see save_traits_collection).'''
    with h5py.File(filename, 'w') as h5:
        h5.create_dataset(name='chosen', data=chosen)


def load_chosen_traits(filename):
    '''save a collection of chosen traits from h5.
    returned as a numpy array of indices in a traits collection)'''
    with h5py.File(filename, 'r') as h5:
        chosen = np.array(h5['chosen'])
    return chosen


def homogeneous_gradient_trait_x(v0, v1, dimsize):
    '''synthetically create an homogeneous gradient between two traits along the X axis'''

    # TODO use linspace
    gradient = np.arange(0, 1, 0.01)
    probs_up = zip([v0] * len(gradient), gradient)
    probs_down = zip([v1] * len(gradient), 1 - gradient)
    probabilities = [dict(combination) for combination in zip(probs_up, probs_down)]

    idx = np.repeat(np.arange(len(probabilities)),
                    np.ceil(float(dimsize[0]) / len(probabilities)))[0: dimsize[0]]

    field = np.ones(dimsize, dtype=np.int) * idx[:, np.newaxis, np.newaxis]
    return probabilities, field


def assign_from_spatial_distribution(positions, spatial_dist, voxel_dimensions):
    '''for every cell in positions, chooses a property from a spatial distribution
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        spatial_dist: a spatial distribution of traits
        voxel_dimensions: the size of the voxels in each dimension
    Returns:
        a list with the same length as positions where each value comes from
        those proposed as keys in the probabilities distributions.
    '''
    voxel_idx = gb.cell_positions_to_voxel_indices(positions, voxel_dimensions)

    voxel_idx_tuple = tuple(voxel_idx.transpose())
    dist_idx_per_position = spatial_dist.field[voxel_idx_tuple]

    L.debug('%d total positions in unknown areas', np.count_nonzero(dist_idx_per_position == -1))

    dist_indices = np.unique(dist_idx_per_position)

    chosen_trait_indices = np.ones_like(dist_idx_per_position) * -1
    for dist_idx in dist_indices:
        if dist_idx != -1 and spatial_dist.distributions[dist_idx]:
            hit_count = np.count_nonzero(dist_idx_per_position == dist_idx)
            trait_indices, trait_probabilities = zip(*spatial_dist.distributions[dist_idx].items())
            chosen = np.random.choice(trait_indices, hit_count, p=trait_probabilities)
            chosen_trait_indices[dist_idx_per_position == dist_idx] = chosen

    return chosen_trait_indices


# TODO review when we use the terms "probabilities", "distribution", etc and be more consitent
# A distribution "assigns a probability to each measurable subset of
# possible outcomes of a random experiment"
# and here is expressed as: {'a': 0.75, 'b': 0.25}
# A distribution_collection is a group of different distributions, probably referenced by a
# volume dataset
# and here is expressed as: [{'a': 0.75, 'b': 0.25}, {'a': 0.5, 'b': 0.5}]
def normalize_probability_distribution(dist):
    '''take a probability distribution for a set of keys and normalize them'''
    total = float(sum(dist.values()))
    return dict((key, p / total) for key, p in dist.items())


def normalize_distribution_collection(distribution_collection):
    '''take a collection of probability distributions and normalize them'''
    return [normalize_probability_distribution(dist) for dist in distribution_collection]


def split_distribution_collection(spatial_dist, attribute):
    '''split a distribution in two or more so that each one only references
    traits with the same value of attribute.
    Each resulting distribution is renormalised.

    this may be generating distributions that are empty but that should not be a problem

    Note that because for every distribution we are creating a new one,
    the indexes of any associated field are still valid.

    returns a dictionary where the keys are all of the possible values of attribute in the
    traits_collection and the values are the resulting distributions.
    '''

    values = set(t[attribute] for t in spatial_dist.traits)

    grouped_distributions = dict((value,
                                  SpatialDistribution(spatial_dist.field,
                                                      [],
                                                      spatial_dist.traits))
                                 for value in values)

    for value in values:
        for distribution in spatial_dist.distributions:

            new_dist = dict((trait_idx, prob)
                            for trait_idx, prob in distribution.iteritems()
                            if spatial_dist.traits[trait_idx][attribute] == value)

            grouped_distributions[value].distributions.append(
                normalize_probability_distribution(new_dist))

    return grouped_distributions


def reduce_distribution_collection(spatial_dist, attribute):
    '''given a spatial distribution, extract an equivalent one where all of the properties
    of the traits collection have been removed except for a specific one.

    For example:

        Taking the traits collection:
            [{'sclass': 'inhibitory', 'mtype': 'Pyramidal'},
             {'sclass': 'inhibitory', 'mtype': 'Martinotti'},
             {'sclass': 'excitatory', 'mtype': 'Martinotti'}]

        And the distributions collection:
            [{0: 0.2,
              1: 0.4,
              2: 0.4}]

        Ignoring all properties except 'sclass' would give the simplified traits collection:
            [{'sclass': 'inhibitory',
             {'sclass': 'excitatory'}]

        And the distributions collection:
            [{0: 0.6,
              1: 0.4}]

    Note that because for every distribution we are creating a new one,
    the indexes of any associated field are still valid.

    Returns:
        a reduced SpatialDistribution
    '''

    # trying to preserve ordering for easy testing
    values = OrderedDict()
    for t in spatial_dist.traits:
        if t[attribute] not in values:
            values[t[attribute]] = len(values)

    def reduce_distribution(d):
        '''reduce a single distribution'''
        new_dist = defaultdict(float)
        for trait_idx, prob in d.iteritems():
            val = spatial_dist.traits[trait_idx][attribute]
            new_trait_idx = values[val]
            new_dist[new_trait_idx] += prob

        return normalize_probability_distribution(new_dist)

    return SpatialDistribution(
        field=spatial_dist.field,
        distributions=[reduce_distribution(distribution)
                       for distribution in spatial_dist.distributions],
        traits=[{attribute: v}
                for v in values.keys()])
