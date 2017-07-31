""" Basic algorithm to create cell positions. """

import numpy as np


def create_cell_positions(density):
    ''' Given cell density volumetric data, create cell positions.

    Cell count is calculated as the sum of cell density values.

    Args:
        density: VoxelData object representing cell density.
            Each voxel contains expected number of cells appearing in this voxel.

    Returns:
        positions: numpy.array of shape (cell_count, 3) where each row represents
            a cell and the columns correspond to (x, y, z).
    '''
    cell_count = int(np.round(np.sum(density.raw)))
    assert cell_count > 0

    voxel_ijk = np.nonzero(density.raw > 0)
    voxel_count = len(voxel_ijk[0])

    probs = 1.0 * density.raw[voxel_ijk] / cell_count
    chosen = np.random.choice(np.arange(voxel_count), cell_count, replace=True, p=probs)
    chosen_idx = np.stack(voxel_ijk).transpose()[chosen]

    # get random positions within chosen voxels
    return density.indices_to_positions(
        chosen_idx + np.random.random(np.shape(chosen_idx))
    )
