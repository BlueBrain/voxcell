""" Basic algorithm to create cell positions. """

import numpy as np


def create_cell_positions(density, density_factor=1.0):
    ''' Given cell density volumetric data, create cell positions.

    Total cell count is calculated based on cell density values.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all voxels

    Returns:
        positions: numpy.array of shape (cell_count, 3) where each row represents
            a cell and the columns correspond to (x, y, z).
    '''
    voxel_mm3 = density.voxel_volume / 1e9  # voxel volume is in um^3
    cell_count_per_voxel = density.raw * density_factor * voxel_mm3
    cell_count = int(np.round(np.sum(cell_count_per_voxel)))
    assert cell_count > 0

    voxel_ijk = np.nonzero(cell_count_per_voxel > 0)
    voxel_idx = np.arange(len(voxel_ijk[0]))

    probs = 1.0 * cell_count_per_voxel[voxel_ijk] / np.sum(cell_count_per_voxel)
    chosen = np.random.choice(voxel_idx, cell_count, replace=True, p=probs)
    chosen_idx = np.stack(voxel_ijk).transpose()[chosen]

    # get random positions within chosen voxels
    return density.indices_to_positions(
        chosen_idx + np.random.random(np.shape(chosen_idx))
    )
