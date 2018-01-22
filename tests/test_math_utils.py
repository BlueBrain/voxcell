import numpy as np
import numpy.testing as npt

import nose.tools as nt

import voxcell.math_utils as test_module


def test_clip():
    r = test_module.clip(
        np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        (np.array([1, 1]), np.array([1, 1]))
    )
    npt.assert_equal(r, np.array([[1]]))


def test_is_diagonal_true():
    A = np.array([
        [2, 0],
        [0, 3]
    ])
    nt.assert_true(test_module.is_diagonal(A))


def test_is_diagonal_false():
    A = np.array([
        [2, 0],
        [1, 3]
    ])
    nt.assert_false(test_module.is_diagonal(A))


def test_lcmm():
    npt.assert_equal(12, test_module.lcmm([2, 3, 4]))


def test_angles_to_matrices_1():
    angles = [np.pi / 2]
    expected = [[
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'x')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_2():
    angles = [np.pi / 2]
    expected = [[
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'y')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_3():
    angles = [np.pi / 2]
    expected = [[
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ]]
    result = test_module.angles_to_matrices(angles, 'z')
    npt.assert_almost_equal(expected, result)


def test_normalize_empty():
    npt.assert_equal(test_module.normalize([]), [])


def test_normalize_0():
    npt.assert_equal(test_module.normalize([1, 0, 0]), [1, 0, 0])


def test_normalize_1():
    npt.assert_equal(test_module.normalize([2, 2, 1]), [2./3, 2./3, 1./3])


def test_normalize_3():
    npt.assert_equal(test_module.normalize([[1, 0, 0], [0, 0, 0]]), [[1, 0, 0], [0, 0, 0]])
