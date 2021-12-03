import numpy as np
import numpy.testing as npt
import pytest

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
    assert test_module.is_diagonal(A)


def test_is_diagonal_false():
    A = np.array([
        [2, 0],
        [1, 3]
    ])
    assert not test_module.is_diagonal(A)


def test_lcmm():
    npt.assert_equal(12, test_module.lcmm([2, 3, 4]))


def test_angles_to_matrices_x():
    angles = [np.pi / 2]
    expected = [[
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'x')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_y():
    angles = [np.pi / 2]
    expected = [[
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'y')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_z():
    angles = [np.pi / 2]
    expected = [[
        [0, -1, 0],
        [1, 0, 0],
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


def test_isin_empty():
    npt.assert_equal(test_module.isin([], [1, 2]), [])


def test_isin_0():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            []
        ),
        [[False, False], [False, False]]
    )


def test_isin_1():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            [2, 4, 4]
        ),
        [[False, True], [True, False]]
    )


def test_isin_2():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            set([1, 2, 3, 5])
        ),
        [[True, True], [True, True]]
    )


def test_euler2mat():
    pi2 = np.pi / 2
    pi3 = np.pi / 3
    pi4 = np.pi / 4
    pi6 = np.pi / 6
    actual = test_module.euler2mat(
        [0.0, pi2, pi3],  # rotation_angle_z
        [pi2, 0.0, pi4],  # rotation_angle_y
        [pi2, pi2, pi6],  # rotation_angle_x
    )
    expected = np.array([
        [
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ],
        [
            [0., -1.,  0.],
            [0.,  0., -1.],
            [1.,  0.,  0.],
        ],
        [
            [0.35355339, -0.61237244,  0.70710678],
            [0.92677670,  0.12682648, -0.35355339],
            [0.12682648,  0.78033009,  0.61237244],
        ]
    ])
    npt.assert_almost_equal(actual, expected)


def test_mat2euler_roundtrip():
    original = np.asarray([
        [
            # ay = pi / 2
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ],
        [
            # ay = -pi / 2
            [ 0., 0., -1.],
            [ 0., 1.,  0.],
            [ 1., 0.,  0.],
        ],
        [
            # ay = pi / 4
            [0.35355339, -0.61237244,  0.70710678],
            [0.92677670,  0.12682648, -0.35355339],
            [0.12682648,  0.78033009,  0.61237244],
        ]
    ])
    actual = test_module.euler2mat(
        *test_module.mat2euler(original)
    )
    npt.assert_almost_equal(original, actual)


def _random_angles(n):
    return 2 * np.pi * np.random.random(n)


def test_mat2euler_roundtrip_random():
    n = 100
    az = _random_angles(n)
    ay = _random_angles(n)
    ax = _random_angles(n)
    mm = test_module.euler2mat(az, ay, ax)
    actual = test_module.euler2mat(
        *test_module.mat2euler(mm)
    )


def test_mat2euler_raises_1():
    with pytest.raises(AssertionError):
        test_module.mat2euler(np.asarray([
            [0., -1.,  0.],
            [0.,  0., -1.],
            [1.,  0.,  0.],
        ]))


def test_mat2euler_raises_2():
    with pytest.raises(AssertionError):
        test_module.mat2euler(np.asarray([
            [
                [0., -1.],
                [1.,  0.],
            ]
        ]))