import numpy as np
import scipy.spatial.distance as distance
import nose.tools as nt

from voxcell import VoxelData

import brainbuilder.cell_positions as test_module


def test_create_cell_positions_1():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    result = test_module.create_cell_positions(density)
    nt.assert_equal(np.shape(result), (27, 3))
    nt.assert_true(np.all((result >= 0) & (result <= 3 * 100)))


def test_create_cell_positions_2():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    result = test_module.create_cell_positions(density, density_factor=0.2)
    nt.assert_equal(np.shape(result), (5, 3))


def test_create_equidistributed_cell_positions_1():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    max_expected_nb_points = 27

    result = test_module.create_cell_positions(density, method='poisson_disc')

    nt.assert_less_equal(result.shape[0], max_expected_nb_points)
    nt.assert_true(np.all((result >= 0) & (result <= 3 * 100)))

    min_distance = 0.84 * 100
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)


def test_create_equidistributed_cell_positions_2():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    method = 'poisson_disc'
    max_expected_nb_points = 5

    result = test_module.create_cell_positions(density, density_factor=0.2,
                                               method='poisson_disc')

    nt.assert_less_equal(result.shape[0], max_expected_nb_points)
    nt.assert_true(np.all((result >= 0) & (result <= 3 * 100)))

    min_distance = 0.84 * 100
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)


def test_create_cell_positions_black_white():
    density = VoxelData(np.zeros((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    density.raw[1, 1, 1] = 27000
    max_expected_nb_points = 27

    result = test_module.create_cell_positions(density, method='poisson_disc')

    nt.assert_less_equal(result.shape[0], max_expected_nb_points)
    nt.assert_true(np.all((result >= 100) & (result <= 2 * 100)))

    min_distance = 0.84 * 100. / np.power(max_expected_nb_points, 1. / 3.)
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)


def test_create_cell_positions_black_grey():
    density = VoxelData(1000 * np.ones((3, 3, 3)), voxel_dimensions=(100, 100, 100))
    density.raw[1, 1, 1] = 4000
    max_expected_nb_points = 30

    result = test_module.create_cell_positions(density, method='poisson_disc')

    nt.assert_less_equal(result.shape[0], max_expected_nb_points)
    nt.assert_true(np.all((result >= 0) & (result <= 3 * 100)))
    nt.assert_false(np.all((result >= 100) & (result <= 2 * 100)))

    # max expected nb points in middle voxel: 4
    min_distance = 0.84 * 100. / np.power(4., 1. / 3.)
    min_distance_between_pts = np.min(distance.pdist(result).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)


def test_get_bbox_indices_nonzero_entries():
    data = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                     [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    bbox = np.array([[1, 1, 2], [3, 2, 3]])

    result = test_module.get_bbox_indices_nonzero_entries(data)

    nt.assert_true(np.array_equal(result, bbox))


def test_get_bbox_nonzero_entries():
    data = np.array([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                     [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    bbox = np.array([[0., 0., 0.], [50., 15., -4.]])
    bbox_nonzero = np.array([[10., 5., -2.], [40., 15., -4.]])
    voxel_dimensions = np.array([10., 5., -1.])

    result = test_module.get_bbox_nonzero_entries(data, bbox, voxel_dimensions)

    nt.assert_true(np.array_equal(result, bbox_nonzero))
