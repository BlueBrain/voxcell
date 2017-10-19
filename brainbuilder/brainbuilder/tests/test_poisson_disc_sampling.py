import numpy as np
import scipy.spatial.distance as distance
import nose.tools as nt

from voxcell import VoxelData

import brainbuilder.poisson_disc_sampling as test_module


def setup_func():
    np.random.seed(42)


def teardown_func():
    pass


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
