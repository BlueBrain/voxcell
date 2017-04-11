import voxcell.math_utils as test_module

import numpy as np
from numpy.testing import assert_equal
from nose.tools import assert_true, assert_false


def test_clip():
    r = test_module.clip(
        np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        (np.array([1, 1]), np.array([1, 1]))
    )
    assert_equal(r, np.array([[1]]))


def test_is_diagonal_true():
    A = np.array([
        [2, 0],
        [0, 3]
    ])
    assert_true(test_module.is_diagonal(A))


def test_is_diagonal_false():
    A = np.array([
        [2, 0],
        [1, 3]
    ])
    assert_false(test_module.is_diagonal(A))


def test_lcmm():
    assert_equal(12, test_module.lcmm([2, 3, 4]))
