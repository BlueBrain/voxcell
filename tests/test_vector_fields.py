from voxcell import vector_fields as vf
import numpy as np
from numpy.testing import assert_array_equal


def test_combine_vector_fields_empty():
    assert_array_equal(vf.combine_vector_fields([]), np.empty((0,)))


def test_combine_vector_fields_low_dim_2():
    r = vf.combine_vector_fields([np.ones((2, 2)) * 1,
                                  np.ones((2, 2)) * 2])

    assert r.shape == (2, 2, 2)


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

    assert r.shape == (2, 2, 2, 3, 3)

    assert_array_equal(r[1, 1, 1], np.array([[1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1]]))

    assert_array_equal(np.dot(r[0, 0, 0], np.array([1, 0, 0])), np.array([1, 0, 0]))


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

    assert r.shape == (2, 2, 2, 3, 3)

    assert_array_equal(r[1, 1, 1], np.array([[0, 0, 1],
                                             [0, 1, 0],
                                             [-1, 0, 0]]))

    assert_array_equal(np.dot(r[1, 1, 1], np.array([1, 0, 0])), np.array([0, 0, -1]))


def test_combine_vector_fields_high_dim():
    r = vf.combine_vector_fields([np.ones((2, 2, 2, 2, 4)) * 1,
                                  np.ones((2, 2, 2, 2, 4)) * 2,
                                  np.ones((2, 2, 2, 2, 4)) * 3,
                                  np.ones((2, 2, 2, 2, 4)) * 4])

    assert r.shape == (2, 2, 2, 2, 4, 4)


def test_join_vector_fields_1_empty():
    assert (vf.join_vector_fields(np.array([[0]])) ==
                 np.array([[0]]))


def test_join_vector_fields_1_full():
    assert (vf.join_vector_fields(np.array([[1]])) ==
                 np.array([[1]]))


def test_join_vector_fields_2():
    assert (vf.join_vector_fields(np.array([[1]]), np.array([[0]])) ==
                 np.array([[1]]))

    assert (vf.join_vector_fields(np.array([[0]]), np.array([[1]])) ==
                 np.array([[1]]))


def test_join_vector_fields_2_override():
    assert (vf.join_vector_fields(np.array([[1]]), np.array([[2]])) ==
                 np.array([[2]]))


def test_gaussian_filter_0():
    assert_array_equal(vf.gaussian_filter(np.array([[1, 0, 0], [1, 0, 0]]), 1),
                       np.array([[1, 0, 0], [1, 0, 0]]))


def test_gaussian_filter_1():
    assert_array_equal(vf.gaussian_filter(np.array([[10, 0, 0], [0, 10, 0]]), 5),
                       np.array([[5, 4, 0], [4, 5, 0]]))
