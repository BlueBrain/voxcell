"""Helper mathematical functions."""

import functools
import math

import numpy as np
from scipy.spatial.transform import Rotation


def gcd(a, b):
    """Return greatest common divisor."""
    return math.gcd(a, b)


def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)


# TODO consider making this a np ufunc
def lcmm(args):
    """Return lcm of args."""
    return functools.reduce(lcm, args)


def minimum_aabb(mask):
    """Calculate the minimum axis-aligned bounding box for a volume mask.

    Returns:
        A tuple containing the minimum x,y,z and maximum x,y,z
    """
    idx = np.nonzero(mask)
    return np.min(idx, axis=1), np.max(idx, axis=1)


def positions_minimum_aabb(positions):
    """Calculate the minimum axis-aligned bounding box for a list of positions.

    Returns:
        A tuple containing the minimum x,y,z and maximum x,y,z
    """
    return np.min(positions, axis=0), np.max(positions, axis=0)


def clip(mask, aabb):
    """Take a numpy array and clip it to an axis-aligned bounding box.

    Args:
        mask: numpy array
        aabb: tuple of two sets of coordinates indicating, respectively,
            the lowest and highest values for each dimension

    Returns:
        A new numpy array containing the same values as mask for the space defined by aabb
    """
    idx = tuple(slice(s, e + 1) for s, e in zip(*aabb))
    return mask[idx].copy()


def is_diagonal(matrix):
    """Check if the matrix is diagonal."""
    return np.all(matrix == np.diag(matrix.diagonal()))


def angles_to_matrices(angles, axis):
    """Convert rotation angles around `axis` to 3x3 rotation matrices.

    Args:
        angles: (N,) array of rotation angles (radian)
        axis: one of ('x', 'y', 'z')

    Returns:
        (N, 3, 3) array of rotation matrices.
    """
    return Rotation.from_euler(axis, angles).as_matrix()


def normalize(vs):
    """Normalize array along last axis."""
    norm = np.linalg.norm(vs, axis=-1)
    norm = np.where(norm > 0, norm, 1.0)
    return vs / norm[..., np.newaxis]


def isin(a, values):
    """Naive NumPy.isin analogue.

    For our usecases (>10^9 non-unique elements in `a`, <10^2 unique elements in tested `values`),
    NumPy.isin() takes same amount of time, but is 3x more memory-hungry.
    """
    a = np.asarray(a)
    result = np.full_like(a, False, dtype=bool)
    for v in set(values):
        result |= (a == v)
    return result


def euler2mat(az, ay, ax):
    """Build 3x3 rotation matrices from az, ay, ax rotation angles (in that order).

    Args:
        az: rotation angles around Z (Nx1 NumPy array; radians)
        ay: rotation angles around Y (Nx1 NumPy array; radians)
        ax: rotation angles around X (Nx1 NumPy array; radians)

    Returns:
        List with 3x3 rotation matrices corresponding to each of N angle triplets.

    See Also:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix (R = X1 * Y2 * Z3)
    """
    c1, s1 = np.cos(ax), np.sin(ax)
    c2, s2 = np.cos(ay), np.sin(ay)
    c3, s3 = np.cos(az), np.sin(az)

    mm = np.array([
        [c2 * c3, -c2 * s3, s2],
        [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
        [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
    ])

    return np.asarray([mm[..., i] for i in range(len(az))])


def mat2euler(mm):
    """Decompose 3x3 rotation matrices into az, ay, ax rotation angles (in that order).

    Args:
        List with 3x3 rotation matrices.

    Returns:
        az: rotation angles around Z (Nx1 NumPy array; radians)
        ay: rotation angles around Y (Nx1 NumPy array; radians)
        ax: rotation angles around X (Nx1 NumPy array; radians)

    See Also:
        https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix (R = X1 * Y2 * Z3)
    """
    assert len(mm.shape) == 3
    assert tuple(mm.shape[1:]) == (3, 3)

    az = np.full(mm.shape[0], np.nan)
    ay = np.full(mm.shape[0], np.nan)
    ax = np.full(mm.shape[0], np.nan)

    sin_ay = mm[:, 0, 2]

    mask1 = np.isclose(sin_ay, 1.0)
    az[mask1] = 0.0
    ay[mask1] = np.pi / 2
    ax[mask1] = np.arctan2(mm[:, 1, 0][mask1], mm[:, 1, 1][mask1])

    mask2 = np.isclose(sin_ay, -1.0)
    az[mask2] = 0.0
    ay[mask2] = -np.pi / 2
    ax[mask2] = np.arctan2(-mm[:, 1, 0][mask2], mm[:, 1, 1][mask2])

    mask3 = np.logical_not(np.logical_or(mask1, mask2))
    az[mask3] = np.arctan2(-mm[:, 0, 1][mask3], mm[:, 0, 0][mask3])
    ay[mask3] = np.arcsin(sin_ay[mask3])
    ax[mask3] = np.arctan2(-mm[:, 1, 2][mask3], mm[:, 2, 2][mask3])

    assert not np.any(np.isnan(az))
    assert not np.any(np.isnan(ay))
    assert not np.any(np.isnan(ax))

    return (az, ay, ax)
