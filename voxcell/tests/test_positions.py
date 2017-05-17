import numpy as np
from nose.tools import eq_
from numpy.testing import assert_equal

from voxcell import positions
from voxcell.voxel_data import VoxelData


def test_cell_counts_to_cell_voxel_indices_0():
    assert_equal(positions._cell_counts_to_cell_voxel_indices(np.zeros((3, 3, 3), dtype=np.int32)),
                 np.empty(shape=(0, 3), dtype=np.int64))


def test_cell_counts_to_cell_voxel_indices_1():
    assert_equal(positions._cell_counts_to_cell_voxel_indices(np.ones((3, 3, 3), dtype=np.int32)),
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
    assert_equal(positions._cell_counts_to_cell_voxel_indices(original),
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [1, 1, 1],
                           [2, 2, 2],
                           [2, 2, 2],
                           [2, 2, 2]], dtype=np.int64))


def test_create_cell_counts_homogeneous_0():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = 0

    result = positions.create_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)

    assert_equal(result, np.zeros(np.shape(result)))


def test_create_cell_counts_homogeneous_1():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = 1

    result = positions.create_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)


def test_create_cell_counts_homogeneous_full():
    cell_body_density_raw = np.ones((3, 3, 3), dtype=np.int)
    total_cell_count = (3 * 3 * 3)

    result = positions.create_cell_counts(cell_body_density_raw, total_cell_count)
    eq_(np.shape(result), np.shape(cell_body_density_raw))
    eq_(np.sum(result), total_cell_count)
#    assert_equal(result, np.ones(np.shape(result)))


mhd = {'voxel_dimensions': (25, 25, 25)}


def test_create_cell_positions_0():
    raw = np.ones((3, 3, 3))
    total_cell_count = 0

    result = positions.create_cell_positions(VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))


def test_create_cell_positions_1():
    raw = np.ones((3, 3, 3))
    total_cell_count = 1

    result = positions.create_cell_positions(VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))


def test_create_cell_positions_full():
    raw = np.ones((3, 3, 3))
    total_cell_count = 3 * 3 * 3

    result = positions.create_cell_positions(VoxelData(raw, **mhd), total_cell_count)

    eq_(np.shape(result), (total_cell_count, 3))
    assert np.all((result >= 0) & (result <= 3 * 25))
