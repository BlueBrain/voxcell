'''algorithm to assign a synapse class to a group of cells'''

import numpy as np


def assign_synapse_class(positions, inhibitory_proportion):
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
