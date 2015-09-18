'''algorithm to assign a synapse class to a group of cells'''
import numpy as np
import pandas as pd


from brainbuilder.utils import traits as tt


def assign_synapse_class_randomly(positions, inhibitory_fraction):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron

    Args:
        positions: list of positions for soma centers (x, y, z).
        inhibitory_fraction: float [0, 1] fraction of cells that will be tagged as Inhibitory.

    Returns:
        A pandas DataFrame with one row for each position and one column: sclass.
        For those positions whose morphology could not be determined, nan is used.
    '''
    chosen = np.random.choice(np.array(['excitatory', 'inhibitory']),
                              size=positions.shape[0],
                              p=np.array([1.0 - inhibitory_fraction, inhibitory_fraction]))

    return pd.DataFrame({'synapse_class': chosen})


def assign_synapse_class_from_spatial_dist(positions, sdist):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron

    Args:
        positions: list of positions for soma centers (x, y, z).
        sdist: SpatialDistribution with at least the property: sClass
        voxel_dimensions: tuple with the size of the voxels in microns in each axis

    Returns:
        A pandas DataFrame with one row for each position and one column: sClass.
        For those positions whose morphology could not be determined, nan is used.
    '''
    chosen = tt.assign_from_spatial_distribution(positions, sdist)
    return pd.DataFrame({'synapse_class': sdist.traits['synapse_class'][chosen].as_matrix()})
