import nose.tools as nt
import numpy as np
import numpy.testing as npt

import brainbuilder.cell_orientations as test_module


def test_apply_rotation_1():
    A = np.random.random((2, 3, 3))
    angles = np.array([0, 0])
    npt.assert_almost_equal(test_module.apply_rotation(A, angles, 'x'), A)
    npt.assert_almost_equal(test_module.apply_rotation(A, angles, 'y'), A)
    npt.assert_almost_equal(test_module.apply_rotation(A, angles, 'z'), A)


def test_apply_rotation_2():
    A = np.array([[
        [ 0.93890941, -0.06521845,  0.33792851],
        [ 0.27854670,  0.72069502, -0.63483102],
        [-0.20214069,  0.69017769,  0.69483372],
    ]])
    angles = np.array([np.pi / 3])
    expected = np.array([[
        [ 0.17680003, -0.06521845,  0.98208365],
        [ 0.68905314,  0.72069502, -0.07618700],
        [-0.70281400,  0.69017769,  0.17235788],
    ]])
    result = test_module.apply_rotation(A, angles, 'y')
    npt.assert_almost_equal(result, expected)


def test_apply_random_rotation():
    A = np.random.random((2, 3, 3))
    nt.assert_equal(test_module.apply_random_rotation(A, 'x').shape, (2, 3, 3))
