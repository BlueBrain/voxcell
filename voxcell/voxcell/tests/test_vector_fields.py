from nose.tools import eq_

from numpy.testing import assert_equal
from voxcell import vector_fields as vf
import numpy as np


def test_generate_homogeneous_field_empty():
    assert_equal(vf.generate_homogeneous_field(np.array([], dtype=np.bool),
                                               np.array([0, 1, 0])),
                 np.empty(shape=(0, 3), dtype=np.int))


def test_generate_homogeneous_field_simple_0():
    assert_equal(vf.generate_homogeneous_field(np.ones((2, 2), dtype=np.bool),
                                               np.array([0, 1, 0])),
                 np.array([[[0, 1, 0], [0, 1, 0]],
                           [[0, 1, 0], [0, 1, 0]]]))


def test_generate_homogeneous_field_simple_1():
    assert_equal(vf.generate_homogeneous_field(np.zeros((2, 2), dtype=np.bool),
                                               np.array([0, 1, 0])),
                 np.array([[[0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0]]]))


def test_combine_vector_fields_empty():
    assert_equal(vf.combine_vector_fields([]), np.empty((0,)))


def test_combine_vector_fields_low_dim_2():
    r = vf.combine_vector_fields([np.ones((2, 2)) * 1,
                                  np.ones((2, 2)) * 2])

    eq_(r.shape, (2, 2, 2))


def test_combine_vector_fields_simple_0():
    # no rotation
    right = [1, 0, 0]
    up = [0, 1, 0]
    fwd = [0, 0, 1]

    r = vf.combine_vector_fields([
            np.array(right * np.prod((2, 2, 2))).reshape((2, 2, 2, 3)),
            np.array(up * np.prod((2, 2, 2))).reshape((2, 2, 2, 3)),
            np.array(fwd * np.prod((2, 2, 2))).reshape((2, 2, 2, 3))
        ])

    eq_(r.shape, (2, 2, 2, 3, 3))

    assert_equal(r[1, 1, 1], np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]]))

    assert_equal(np.dot(r[0, 0, 0], np.array([1, 0, 0])), np.array([1, 0, 0]))


def test_combine_vector_fields_simple_1():
    # 90 degree rotation around Y
    right = [0, 0, -1]
    up = [0, 1, 0]
    fwd = [1, 0, 0]

    r = vf.combine_vector_fields([
            np.array(right * np.prod((2, 2, 2))).reshape((2, 2, 2, 3)),
            np.array(up * np.prod((2, 2, 2))).reshape((2, 2, 2, 3)),
            np.array(fwd * np.prod((2, 2, 2))).reshape((2, 2, 2, 3))
        ])

    eq_(r.shape, (2, 2, 2, 3, 3))

    assert_equal(r[1, 1, 1], np.array([[0, 0, 1],
                                       [0, 1, 0],
                                       [-1, 0, 0]]))

    assert_equal(np.dot(r[1, 1, 1], np.array([1, 0, 0])), np.array([0, 0, -1]))


def test_combine_vector_fields_high_dim():
    r = vf.combine_vector_fields([np.ones((2, 2, 2, 2, 4)) * 1,
                                  np.ones((2, 2, 2, 2, 4)) * 2,
                                  np.ones((2, 2, 2, 2, 4)) * 3,
                                  np.ones((2, 2, 2, 2, 4)) * 4])

    eq_(r.shape, (2, 2, 2, 2, 4, 4))


def test_join_vector_fields_1_empty():
    assert_equal(vf.join_vector_fields(np.array([[0]])),
                 np.array([[0]]))


def test_join_vector_fields_1_full():
    assert_equal(vf.join_vector_fields(np.array([[1]])),
                 np.array([[1]]))


def test_join_vector_fields_2():
    assert_equal(vf.join_vector_fields(np.array([[1]]), np.array([[0]])),
                 np.array([[1]]))

    assert_equal(vf.join_vector_fields(np.array([[0]]), np.array([[1]])),
                 np.array([[1]]))


def test_join_vector_fields_2_override():
    assert_equal(vf.join_vector_fields(np.array([[1]]), np.array([[2]])),
                 np.array([[2]]))


def test_calculate_fields_by_distance_from_1d():
    reg = np.array([1, 1, 1, 1], dtype=np.bool)
    ref = np.array([0, 0, 1, 1], dtype=np.bool)
    assert_equal(
        vf.calculate_fields_by_distance_from(reg, ref),
        np.array([[-1], [-1], [-0.5], [0]]))


def test_calculate_fields_by_distance_to_1d():
    reg = np.array([1, 1, 1, 1], dtype=np.bool)
    ref = np.array([0, 0, 1, 1], dtype=np.bool)
    assert_equal(
        vf.calculate_fields_by_distance_to(reg, ref),
        np.array([[1], [1], [0.5], [0]]))


def test_calculate_fields_by_distance_to_between():
    reg = np.array([1, 1, 1, 1], dtype=np.bool)
    ref0 = np.array([1, 1, 0, 0], dtype=np.bool)
    ref1 = np.array([0, 0, 1, 1], dtype=np.bool)
    assert_equal(
        vf.calculate_fields_by_distance_between(reg, ref0, ref1),
        np.array([[1], [0.5], [1], [1]]))  # everything positive: pointing to ref1


def test_gaussian_filter_0():
    assert_equal(vf.gaussian_filter(np.array([[1, 0, 0], [1, 0, 0]]), 1),
                 np.array([[1, 0, 0], [1, 0, 0]]))


def test_gaussian_filter_1():
    assert_equal(vf.gaussian_filter(np.array([[10, 0, 0], [0, 10, 0]]), 5),
                 np.array([[5, 4, 0], [4, 5, 0]]))
