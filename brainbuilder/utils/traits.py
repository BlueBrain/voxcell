'''
    library to handle traits fields and collections and the logic to assign them
'''

import numpy as np
import h5py
import json

from brainbuilder.utils import genbrain as gb


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


def homogeneous_gradient_tait_x(v0, v1, dimsize):
    '''synthetically create an homogeneous gradient between two traits along the X axis'''

    gradient = np.arange(0, 1, 0.01)
    probs_up = zip([v0] * len(gradient), gradient)
    probs_down = zip([v1] * len(gradient), 1 - gradient)
    probabilities = [dict(combination) for combination in zip(probs_up, probs_down)]

    idx = np.repeat(np.arange(len(probabilities)),
                    np.ceil(float(dimsize[0]) / len(probabilities)))[0: dimsize[0]]

    field = np.ones(dimsize, dtype=np.int) * idx[:, np.newaxis, np.newaxis]
    return probabilities, field


def assign_from_spacial_distribution(positions, field, probabilities, voxel_dimensions):
    '''for every cell in positions, chooses a property from a spacial distribution
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        traits_field: volume data where every voxel is an index in the probabilities list
        field: a volume dataset where every voxel contains an index in a list of
            probability distributions
        probabilites: a list of all possible probability distributions for each value
        voxel_dimensions: the size of the voxels in each dimension
    Returns:
        a list of sclass values that correspond to each position
    '''
    voxel_idx = gb.cell_positions_to_voxel_indices(positions, voxel_dimensions)

    voxel_idx_tuple = tuple(voxel_idx.transpose())
    probs_idx_per_position = field[voxel_idx_tuple]

    print np.count_nonzero(probs_idx_per_position == -1), 'total positions in unknown areas'

    probabilities_idx = np.unique(probs_idx_per_position)

    chosen_trait_indices = np.ones_like(probs_idx_per_position) * -1
    for prob_idx in probabilities_idx:
        if prob_idx != -1:
            hit_count = np.count_nonzero(probs_idx_per_position == prob_idx)

            trait_indices, trait_probabilities = zip(*probabilities[prob_idx].items())
            chosen = np.random.choice(trait_indices, hit_count, p=trait_probabilities)
            chosen_trait_indices[probs_idx_per_position == prob_idx] = chosen

    return chosen_trait_indices
