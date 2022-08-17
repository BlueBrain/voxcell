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


def voxel_intersection(line_segment, data, return_sub_segments=False):
    """Find voxels intersected by a given segment and cut the segment according to these voxels.

    Args:
        line_segment: The segment with the following form: [[x0, y0, z0], [x1, y1, z1]].
        data: The VoxelData object.
        return_sub_segments: If true the sub segments are also returned with the voxel indices.

    Returns:
        List of 3D indices.
        If `return_sub_segments` is `True`, the coordinates of the sub-segment points are returned.
    """
    # pylint: disable=too-many-locals

    # If the segment is outside the bounding box, then it does not intersect any voxels.
    if (line_segment < data.bbox).all() or (line_segment >= data.bbox).all():
        seg_point_indices = np.zeros((0, 3), dtype=np.result_type(0))
        if return_sub_segments:
            return seg_point_indices, np.reshape(line_segment, (1, -1))
        return seg_point_indices

    # The segment is clipped inside the global bbox.
    clipped_line_segment = np.clip(
        line_segment,
        a_min=data.bbox[0],
        a_max=np.nextafter(data.bbox[1], data.bbox[1] - 1),
    )

    start_pt, end_pt = clipped_line_segment

    def get_intersections(dst1, dst2):
        """Compute intersection point."""
        same_sign = np.sign(dst1) == np.sign(dst2)
        coplanar = (dst1 == 0) & (dst2 == 0)
        denomimator = dst2 - dst1
        denomimator = np.where(denomimator == 0, np.nan, denomimator)
        f = np.where(same_sign | coplanar, np.nan, -dst1 / denomimator)

        # Return the hit position.
        return start_pt + (end_pt - start_pt) * f[:, np.newaxis]

    # Compute the actual bbox of the segment
    bbox = np.sort(data.positions_to_indices(clipped_line_segment), axis=0)

    # Get the coordinates of the planes between voxels
    planes = [data.offset[i] + np.arange(bbox[0, i], bbox[1, i] + 1) * data.voxel_dimensions[i]
              for i in range(3)]

    # Get the coordinates of the intersection points
    seg_points = np.vstack(
        [get_intersections(start_pt[i] - planes[i], end_pt[i] - planes[i])
         for i in range(3)]
    )

    # Build the sub-segment coordinate array
    seg_points = seg_points[~np.isnan(seg_points).all(axis=1)]
    seg_points = np.unique(seg_points, axis=0)

    # Check how the points are ordered along each axis
    xyz_ascending = np.sign(end_pt - start_pt)

    # Remove duplicated points when the extremities of the segment are on a voxel boundary, except
    # if they are in the ascending quadrant.
    new_pts = [seg_points]
    if xyz_ascending.sum() >= 0:
        new_pts = new_pts + [end_pt]
        seg_pt_start = (seg_points == start_pt).all(axis=1)
        if (
            not seg_pt_start.any()
            and not any(len(planes[i]) > 1 and (start_pt[i] == planes[i]).any() for i in range(3))
        ):
            new_pts = [start_pt] + new_pts
    else:
        new_pts = [start_pt] + new_pts
        seg_pt_end = (seg_points == end_pt).all(axis=1)
        if (
            not seg_pt_end.any()
            and not any(len(planes[i]) > 1 and (end_pt[i] == planes[i]).any() for i in range(3))
        ):
            new_pts = new_pts + [end_pt]

    # Build the sub-segment points
    seg_points = np.vstack(new_pts)

    # Sort the sub-segment points so their order is consistent with the given segment, i.e. the
    # first point is the start point, the second pt is the closest to the start pt, etc.
    ind = np.lexsort((seg_points[:, 0], seg_points[:, 1], seg_points[:, 2]))

    if not (
        xyz_ascending[0] == 1
        or (xyz_ascending[0] == 0 and xyz_ascending[1] == 1)
        or (xyz_ascending[0] == 0 and xyz_ascending[1] == 0 and xyz_ascending[2] == 1)
    ):
        ind = ind[::-1]

    seg_points = seg_points[ind, :]

    # Find the intersection indices
    seg_point_indices = data.positions_to_indices((seg_points[:-1, :] + seg_points[1:, :]) / 2.)

    if return_sub_segments:
        sub_segments = np.hstack([seg_points[:-1], seg_points[1:]])
        return seg_point_indices, sub_segments

    return seg_point_indices
