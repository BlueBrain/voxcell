import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import voxcell.quaternion as test_module


def check_normalized(mat):
    assert_almost_equal(np.dot(mat, mat.T)[0][0], 1.0)


def check_matrices_to_quaternions(count, matrix, quaternion):
    quat = test_module.matrices_to_quaternions(np.array([matrix] * count))
    assert_almost_equal(quat, np.array([quaternion] * count), decimal=6)
    check_normalized(quat)


def check_quaternions_to_matrices(count, matrix, quaternion):
    m = test_module.quaternions_to_matrices(np.array([quaternion] * count))
    assert_almost_equal(m, np.array([matrix] * count), decimal=6)


def test_quaternion_empty():
    quat = test_module.matrices_to_quaternions(np.empty((0, 3, 3)))
    assert_array_equal(quat, np.empty((0, 4)))


def test_quaternion_identity():
    m = np.diag([1, 1, 1])
    q = np.array([0, 0, 0, 1])

    check_matrices_to_quaternions(1, m, q)
    check_matrices_to_quaternions(2, m, q)
    check_matrices_to_quaternions(100, m, q)


def test_quaternion_90_y():
    m = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-1, 0, 0]])

    q = np.array([0., 0.707107, 0., 0.707107])

    check_matrices_to_quaternions(1, m, q)
    check_matrices_to_quaternions(2, m, q)
    check_matrices_to_quaternions(100, m, q)


def test_quaternion_copysign():
    # copysign should not affect our implementaton
    # see:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/paul.htm
    m = np.diag([1, 1, -1])
    q = np.array([0, 0, 0, 1])

    check_matrices_to_quaternions(1, m, q)
    check_matrices_to_quaternions(2, m, q)
    check_matrices_to_quaternions(100, m, q)


def test_quaternion_180_y():
    m = np.diag([-1, 1, -1])
    q = np.array([0, 1, 0, 0])

    check_matrices_to_quaternions(1, m, q)
    check_matrices_to_quaternions(2, m, q)
    check_matrices_to_quaternions(100, m, q)


def test_quaternion_180_heading_90_attitude():
    m = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]])

    q = np.array([0.707107, 0.707107, 0, 0])

    check_matrices_to_quaternions(1, m, q)
    check_matrices_to_quaternions(2, m, q)
    check_matrices_to_quaternions(100, m, q)


def test_matrix_empty():
    quat = test_module.quaternions_to_matrices(np.empty((0, 4)))
    assert_array_equal(quat, np.empty((0, 3, 3)))


def test_matrix_identity():
    m = np.diag([1, 1, 1])
    q = np.array([0, 0, 0, 1])

    check_quaternions_to_matrices(1, m, q)
    check_quaternions_to_matrices(2, m, q)
    check_quaternions_to_matrices(100, m, q)

    # check non-normalized quaternion
    check_quaternions_to_matrices(1, m, 1000 * q)


def test_matrix_90_y():
    m = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [-1, 0, 0]])

    q = np.array([0., 0.707107, 0., 0.707107])

    check_quaternions_to_matrices(1, m, q)
    check_quaternions_to_matrices(2, m, q)
    check_quaternions_to_matrices(100, m, q)

    # check non-normalized quaternion
    check_quaternions_to_matrices(1, m, 1000 * q)


def test_matrix_180_y():
    m = np.diag([-1, 1, -1])
    q = np.array([0, 1, 0, 0])

    check_quaternions_to_matrices(1, m, q)
    check_quaternions_to_matrices(2, m, q)
    check_quaternions_to_matrices(100, m, q)

    # check non-normalized quaternion
    check_quaternions_to_matrices(1, m, 1000 * q)


def test_matrix_180_heading_90_attitude():
    m = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]])

    q = np.array([0.707107, 0.707107, 0, 0])

    check_quaternions_to_matrices(1, m, q)
    check_quaternions_to_matrices(2, m, q)
    check_quaternions_to_matrices(100, m, q)

    # check non-normalized quaternion
    check_quaternions_to_matrices(1, m, 1000 * q)


def test_roundtrip_complex():
    # these are all 24 cube rotations taken from:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/examples/index.htm

    series = np.array([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [0, 1, 0],
         [-1, 0, 0]],
        [[-1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]],
        [[0, 0, -1],
         [0, 1, 0],
         [1, 0, 0]],
        [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]],
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, -1]],
        [[0, 0, -1],
         [1, 0, 0],
         [0, -1, 0]],
        [[0, 1, 0],
         [-1, 0, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [-1, 0, 0],
         [0, -1, 0]],
        [[0, -1, 0],
         [-1, 0, 0],
         [0, 0, -1]],
        [[0, 0, -1],
         [-1, 0, 0],
         [0, 1, 0]],
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]],
        [[0, 1, 0],
         [0, 0, -1],
         [-1, 0, 0]],
        [[-1, 0, 0],
         [0, 0, -1],
         [0, -1, 0]],
        [[0, -1, 0],
         [0, 0, -1],
         [1, 0, 0]],
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]],
        [[0, 0, -1],
         [0, -1, 0],
         [-1, 0, 0]],
        [[-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]],
        [[0, 0, 1],
         [0, -1, 0],
         [1, 0, 0]],
        [[1, 0, 0],
         [0, 0, 1],
         [0, -1, 0]],
        [[0, -1, 0],
         [0, 0, 1],
         [-1, 0, 0]],
        [[-1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]],
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]]],
        dtype=np.float64)

    quat = test_module.matrices_to_quaternions(series)
    restored = test_module.quaternions_to_matrices(quat)
    assert_almost_equal(series, restored, decimal=6)
    check_normalized(quat)
