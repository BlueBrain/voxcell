'''Basic algorithm to create positions for cell bodies from a density distribution'''
import numpy as np
from voxcell import deprecate


def create_cell_counts(density, total_count):
    '''create a matrix of the same dimensions of the cell_body_density where each value
    is an integer with the number of cells expected to ocurr in that volume

    Args:
        density: numpy.array representing cell body density.
            Each voxel represents a value that once normalised, can be treated as a probability of
            cells appearing in this voxel.
        total_count: positive int. Number of cell positions to generate.

    Returns:
        counts: numpy.array of same shape as cell_body_density.
    '''

    all_voxels = np.arange(density.size)

    probs = density.flatten().astype(np.float64)
    probs /= np.sum(probs)

    chosen_voxels = np.random.choice(all_voxels, size=total_count, p=probs)

    unique, counts = np.unique(chosen_voxels, return_counts=True)

    chosen_indexes = np.unravel_index(unique, density.shape)

    assigned = np.zeros_like(density)

    assigned[chosen_indexes] = counts

    assert np.sum(assigned) == total_count

    return assigned.astype(np.int)


def create_cell_positions(density, total_count):
    ''' create a series of cell positions

    Args:
        density: VoxelData object representing cell body density.
            Each voxel represents a value that once normalised, can be treated as a probability of
            cells appearing in this voxel.
        total_count: positive int. Number of cell positions to generate.

    Returns:
        positions: numpy.array of shape (total_cell_count, 3) where each row represents
            a cell and the columns represent (x, y, z).
    '''
    deprecate.warn("Deprecated. Please use 'brainbuilder.cell_positions' module instead.")
    cell_counts_per_voxel = create_cell_counts(density.raw, total_count)

    assert np.sum(cell_counts_per_voxel) == total_count, \
        '%s != %s' % (np.sum(cell_counts_per_voxel), total_count)

    cell_voxel_indices = _cell_counts_to_cell_voxel_indices(cell_counts_per_voxel)

    # get random positions within voxels
    cell_voxel_indices = (
        cell_voxel_indices.astype(np.float32) +
        np.random.random(np.shape(cell_voxel_indices))
    )

    return density.indices_to_positions(cell_voxel_indices)


def _cell_counts_to_cell_voxel_indices(cell_counts_per_voxel):
    '''take a matrix with an element per voxel that represents the number of cells in that space
    and return a matrix with a row for each cell where the values represent its corresponding
    voxel's X Y Z indices'''

    idx = np.nonzero(cell_counts_per_voxel)
    repeats = cell_counts_per_voxel[idx]
    locations = np.repeat(np.array(idx).transpose(), repeats, 0)

    return locations
