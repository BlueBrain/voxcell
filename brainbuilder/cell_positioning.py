'''task that positions cell bodies from a density distribution'''
from brainbuilder.utils import genbrain as gb
from brainbuilder.utils.math import unique_with_counts
import numpy as np

import logging
L = logging.getLogger(__name__)


def assign_cell_counts(cell_body_density_raw, total_cell_count):
    '''create a matrix of the same dimensions of the cell_body_density where each value
    is an integer with the number of cells expected to ocurr in that volume'''

    all_voxels = np.arange(cell_body_density_raw.size)

    probs = cell_body_density_raw.flatten().astype(np.float64)
    probs /= np.sum(probs)

    chosen_voxels = np.random.choice(all_voxels, size=total_cell_count, p=probs)

    unique, counts = unique_with_counts(chosen_voxels)

    chosen_indexes = np.unravel_index(unique, cell_body_density_raw.shape)

    assigned = np.zeros_like(cell_body_density_raw)

    assigned[chosen_indexes] = counts

    assert np.sum(assigned) == total_cell_count

    return assigned.astype(np.int)


def cell_counts_to_cell_voxel_indices(cell_counts_per_voxel):
    '''take a matrix with an element per voxel that represents the number of cells in that space
    and return a matrix with a row for each cell where the values represent its corresponding
    voxel's X Y Z indices'''

    idx = np.nonzero(cell_counts_per_voxel)
    repeats = cell_counts_per_voxel[idx]
    locations = np.repeat(np.array(idx).transpose(), repeats, 0)

    return locations


def cell_positioning(density, total_cell_count):
    '''
    Args:
        density_raw: voxel data from Allen Brain Institute.
            Called "atlasVolume" in their website.
            Each voxel represents a value that once normalised, can be treated as a probability of
            cells appearing in this voxel.
        total_cell_count: an int

    Returns:
        positions: list of positions for soma centers (x, y, z).
    '''

    cell_counts_per_voxel = assign_cell_counts(density.raw, total_cell_count)

    assert np.sum(cell_counts_per_voxel) == total_cell_count, \
        '%s != %s' % (np.sum(cell_counts_per_voxel), total_cell_count)

    cell_voxel_indices = cell_counts_to_cell_voxel_indices(cell_counts_per_voxel)

    positions = gb.cell_voxel_indices_to_positions(cell_voxel_indices,
                                                   density.mhd['ElementSpacing'])

    return positions
