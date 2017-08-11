""" Basic algorithm to create cell positions. """

import numpy as np


def create_cell_positions(density, density_factor=1.0):
    ''' Given cell density volumetric data, create cell positions.

    Cell count is calculated as the sum of cell density values.

    Args:
        density(VoxelData): cell density; expected number of cells appearing in each voxel
        density_factor(float): reduce / increase density proportionally for all voxels

    Returns:
        positions: numpy.array of shape (cell_count, 3) where each row represents
            a cell and the columns correspond to (x, y, z).
    '''
    cell_count = int(np.round(density_factor * np.sum(density.raw)))
    assert cell_count > 0

    voxel_ijk = np.nonzero(density.raw > 0)
    voxel_count = len(voxel_ijk[0])

    probs = 1.0 * density.raw[voxel_ijk] / np.sum(density.raw)
    chosen = np.random.choice(np.arange(voxel_count), cell_count, replace=True, p=probs)
    chosen_idx = np.stack(voxel_ijk).transpose()[chosen]

    # get random positions within chosen voxels
    return density.indices_to_positions(
        chosen_idx + np.random.random(np.shape(chosen_idx))
    )
