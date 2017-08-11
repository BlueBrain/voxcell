import numpy as np
import nose.tools as nt

from voxcell import VoxelData

import brainbuilder.cell_positions as test_module


def test_create_cell_positions_1():
    density = VoxelData(np.ones((3, 3, 3)), voxel_dimensions=(25, 25, 25))
    result = test_module.create_cell_positions(density)
    nt.assert_equal(np.shape(result), (27, 3))
    nt.assert_true(np.all((result >= 0) & (result <= 3 * 25)))


def test_create_cell_positions_2():
    density = VoxelData(np.ones((3, 3, 3)), voxel_dimensions=(25, 25, 25))
    result = test_module.create_cell_positions(density, density_factor=0.2)
    nt.assert_equal(np.shape(result), (5, 3))
