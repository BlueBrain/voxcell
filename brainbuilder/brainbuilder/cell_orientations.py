""" Helper methods for cell orientations. """

import numpy as np

from voxcell.math_utils import angles_to_matrices


def apply_rotation(A, angles, axis):
    """
    Apply rotation around given `axis`.

    Args:
        A: (N, 3, 3) array of rotation matrices
        angles: (N,) array of rotation angles (radian)
        axis: one of ('x', 'y', 'z')

    Returns:
        (N, 3, 3) array of mutated rotation matrices
        (i.e., [A[k] * ROTATE(angles[k], axis) for k in 0..N-1])
    """
    rotations = angles_to_matrices(angles, axis=axis)
    return np.einsum('...ij,...jk->...ik', A, rotations)


def apply_random_rotation(A, axis):
    """
    Apply random rotation around given `axis`.

    Args:
        A: (N, 3, 3) array of rotation matrices
        axis: one of ('x', 'y', 'z')

    Returns:
        (N, 3, 3) array of mutated rotation matrices
    """
    angles = np.pi * np.random.random(A.shape[0])
    return apply_rotation(A, angles, axis)
