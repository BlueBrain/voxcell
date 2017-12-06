import numpy as np
import scipy.spatial.distance as distance
import nose.tools as nt

from voxcell import VoxelData

import brainbuilder.poisson_disc_sampling as test_module
from brainbuilder.exceptions import BrainBuilderError


def setup_func():
    np.random.seed(42)


def teardown_func():
    pass


def test_grid_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # mark some cells as non-empty
    grid.grid[0, 0, 0] = 0
    grid.grid[0, 1, 3] = 0
    grid.grid[0, 2, 4] = 0

    empty_cell = grid.get_random_empty_grid_cell()

    nt.assert_equal(grid.grid[empty_cell], -1)


def test_grid_no_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # no empty cells
    grid.grid = np.ones(grid.grid.shape)

    nt.assert_raises(BrainBuilderError, grid.get_random_empty_grid_cell)


def test_generate_random_point_in_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # mark some cells as non-empty
    grid.grid[0, 0, 0] = 0
    grid.grid[0, 1, 3] = 0
    grid.grid[0, 2, 4] = 0

    point = grid.generate_random_point_in_empty_grid_cell()

    grid_point = grid.get_grid_coords(point)
    nt.assert_equal(grid.grid[grid_point], -1)


def test_generate_random_point_in_empty_cell():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    grid = test_module.Grid(domain, 100)
    # no empty cells
    grid.grid = np.ones(grid.grid.shape)

    nt.assert_raises(BrainBuilderError,
                     grid.generate_random_point_in_empty_grid_cell)


@nt.with_setup(setup_func, teardown_func)
def test_generate_points():
    domain = np.array([[0, 0, 0], [100, 200, 500]])
    nb_points = 20
    min_distance = 5
    seed = np.array([0, 0, 0])
    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func,
                                         seed)

    nt.assert_equal(len(points), nb_points)
    nt.assert_true(np.all(np.equal(seed, points[0])))
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)
    for point in points:
        nt.assert_true(np.all(point >= domain[0,:]) and np.all(point <= domain[1,:]))


@nt.with_setup(setup_func, teardown_func)
def test_generate_points_too_many():
    domain = np.array([[0, 0, 0], [10, 20, 5]])
    nb_points = 1000
    min_distance = 1
    seed = np.array([0, 0, 0])
    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func,
                                         seed)

    nt.assert_less(len(points), nb_points)
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)
    for point in points:
        nt.assert_true(np.all(point >= domain[0,:]) and np.all(point <= domain[1,:]))


@nt.with_setup(setup_func, teardown_func)
def test_generate_points_random_seed():
    domain = np.array([[0, 0, 0], [100, 200, 50]])
    nb_points = 20
    min_distance = 5
    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    nt.assert_equal(len(points), nb_points)
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)
    for point in points:
        nt.assert_true(np.all(point >= domain[0,:]) and np.all(point <= domain[1,:]))


@nt.with_setup(setup_func, teardown_func)
def test_generate_points_random_seed_neg_domain():
    domain = np.array([[-50, -100, -25], [50, 100, 25]])
    nb_points = 20
    min_distance = 5
    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    nt.assert_equal(len(points), nb_points)
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)
    for point in points:
        nt.assert_true(np.all(point >= domain[0,:]) and np.all(point <= domain[1,:]))


@nt.with_setup(setup_func, teardown_func)
def test_generate_points_random_seed_neg_domain_2():
    domain = np.array([[-50, -100, -25], [-150, 100, 25]])
    nb_points = 20
    min_distance = 5
    def min_distance_func(point=None):
        return min_distance

    points = test_module.generate_points(domain, nb_points, min_distance_func)

    nt.assert_equal(len(points), nb_points)
    min_distance_between_pts = np.min(distance.pdist(points).flatten())
    nt.assert_less_equal(min_distance, min_distance_between_pts)
    for point in points:
        nt.assert_true((point[0] <= domain[0,0]) and (point[0] >= domain[1,0]))
        nt.assert_true(np.all(point[1:] >= domain[0,1:]) and np.all(point[1:] <= domain[1,1:]))
