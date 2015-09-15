from numpy.testing import assert_equal
from nose.tools import eq_

from brainbuilder.utils import vector_fields as vf
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
