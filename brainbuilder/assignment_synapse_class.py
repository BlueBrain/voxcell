'''algorithm to assign a synapse class to a group of cells'''

import numpy as np

from brainbuilder.utils import traits as tt
from brainbuilder.utils import bbp


def assign_synapse_class_randomly(positions, inhibitory_proportion):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        proportion: float [0, 1] percentage of cells that will be tagged as Inhibitory.
    Returns:
        a list of synapse class values that correspond to each position
    '''
    return np.random.choice(np.array(['excitatory', 'inhibitory']),
                            size=positions.shape[0],
                            p=np.array([1.0 - inhibitory_proportion, inhibitory_proportion]))


def assign_synapse_class_from_recipe(positions, annotation, hierarchy,
                                     recipe_filename, region_name):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron
    Returns:
        a list of synapse class values that correspond to each position
    '''

    (traits_field, distribution_collection, traits_collection) = \
        bbp.load_recipe_as_spatial_distributions(recipe_filename,
                                                 annotation.raw, hierarchy, region_name)

    sclass_dist_col, sclass_traits_col = tt.reduce_distribution_collection(distribution_collection,
                                                                           traits_collection,
                                                                           'sClass')

    chosen_sclass = tt.assign_from_spatial_distribution(positions,
                                                        traits_field, sclass_dist_col,
                                                        annotation.mhd['ElementSpacing'])

    return [sclass_traits_col[idx]['sClass'] for idx in chosen_sclass]
