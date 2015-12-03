from nose.tools import eq_
import numpy as np

from brainbuilder import cell_positioning
from brainbuilder.utils import core
from numpy.testing import assert_equal

from brainbuilder import cell_positioning as cp


def test_cell_counts_to_cell_voxel_indices_0():
    assert_equal(cp.cell_counts_to_cell_voxel_indices(np.zeros((3, 3, 3), dtype=np.int32)),
                 np.empty(shape=(0, 3), dtype=np.int64))


def test_cell_counts_to_cell_voxel_indices_1():
    assert_equal(cp.cell_counts_to_cell_voxel_indices(np.ones((3, 3, 3), dtype=np.int32)),
                 np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 2],
                          [0, 1, 0],
                          [0, 1, 1],
                          [0, 1, 2],
                          [0, 2, 0],
                          [0, 2, 1],
                          [0, 2, 2],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 0, 2],
                          [1, 1, 0],
                          [1, 1, 1],
                          [1, 1, 2],
                          [1, 2, 0],
                          [1, 2, 1],
                          [1, 2, 2],
                          [2, 0, 0],
                          [2, 0, 1],
                          [2, 0, 2],
                          [2, 1, 0],
                          [2, 1, 1],
                          [2, 1, 2],
                          [2, 2, 0],
                          [2, 2, 1],
                          [2, 2, 2]]))


def test_cell_counts_to_cell_voxel_indices_complex():
    original = np.zeros((3, 3, 3), dtype=np.int32)
    original[0, 0, 0] = 2
    original[1, 1, 1] = 1
    original[2, 2, 2] = 3
    assert_equal(cp.cell_counts_to_cell_voxel_indices(original),
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 1, 1],
                           [2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]], dtype=np.int64))


def test_assign_cell_counts_homogeneous_0():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = 0

    result = cp.assign_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)

    assert_equal(result, np.zeros(np.shape(result)))


def test_assign_cell_counts_homogeneous_1():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = 1

    result = cp.assign_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)


def test_assign_cell_counts_homogeneous_full():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = (3 * 3 * 3)

    result = cp.assign_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)
#    assert_equal(result, np.ones(np.shape(result)))


def test_cell_voxel_indices_to_positions_0():
    cell_voxel_indices = np.zeros((0, 3), dtype=np.int)
    voxel_dimensions = (3, 3, 3)
    result = cell_positioning._cell_voxel_indices_to_positions(cell_voxel_indices, voxel_dimensions)

    eq_(np.shape(result), (0, 3))


def test_cell_voxel_indices_to_positions_1():
    cell_voxel_indices = np.zeros((1, 3), dtype=np.int)
    voxel_dimensions = (0, 1, 100)
    result = cell_positioning._cell_voxel_indices_to_positions(cell_voxel_indices, voxel_dimensions)

    eq_(np.shape(result), (1, 3))
    eq_(result[0, 0], 0)
    assert (result[0, 1] >= 0) and (result[0, 1] <= 1)
    assert (result[0, 2] >= 0) and (result[0, 2] <= 100)


def test_cell_voxel_indices_to_positions_2():
    cell_voxel_indices = np.zeros((2, 3), dtype=np.int)
    cell_voxel_indices[1] = np.ones((1, 3))
    voxel_dimensions = (0, 1, 100)
    result = cell_positioning._cell_voxel_indices_to_positions(cell_voxel_indices, voxel_dimensions)

    eq_(np.shape(result), (2, 3))

    eq_(result[0, 0], 0)
    assert (result[0, 1] >= 0) and (result[0, 1] <= 1)
    assert (result[0, 2] >= 0) and (result[0, 2] <= 100)

    eq_(result[1, 0], 0)
    assert (result[1, 1] >= 1) and (result[1, 1] <= 2)
    assert (result[1, 2] >= 100) and (result[0, 2] <= 200)


mhd = {'voxel_dimensions': (25, 25, 25)}


def test_cell_positioning_0():
    raw = np.ones((3, 3, 3))
    total_cell_count = 0

    result = cp.cell_positioning(core.VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))


def test_cell_positioning_1():
    raw = np.ones((3, 3, 3))
    total_cell_count = 1

    result = cp.cell_positioning(core.VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))


def test_cell_positioning_full():
    raw = np.ones((3, 3, 3))
    total_cell_count = 3 * 3 * 3

    result = cp.cell_positioning(core.VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))
