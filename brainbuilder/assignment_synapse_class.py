'''algorithm to assign a synapse class to a group of cells'''
import numpy as np

from brainbuilder.utils import traits as tt


def assign_synapse_class_randomly(positions, inhibitory_fraction):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron

    Args:
        positions: list of positions for soma centers (x, y, z).
        inhibitory_fraction: float [0, 1] fraction of cells that will be tagged as Inhibitory.

    Returns:
        An array of synapse class values that correspond to each position
    '''
    return np.random.choice(np.array(['excitatory', 'inhibitory']),
                            size=positions.shape[0],
                            p=np.array([1.0 - inhibitory_fraction, inhibitory_fraction]))


def assign_synapse_class_from_spatial_dist(positions, spatial_dist, voxel_dimensions):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron

    Args:
        positions: list of positions for soma centers (x, y, z).
        spatial_dist: SpatialDistribution with at least the property: sClass
        voxel_dimensions: tuple with the size of the voxels in microns in each axis

    Returns:
        An array of synapse class values that correspond to each position
    '''
    chosen_sclass = tt.assign_from_spatial_distribution(positions, spatial_dist, voxel_dimensions)
    return spatial_dist.traits['sClass'][chosen_sclass].as_matrix()
