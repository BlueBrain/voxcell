'''functions to build artificial shapes'''

import numpy as np

from voxcell import math_utils
from voxcell.voxel_data import VoxelData

from voxcell.utils import deprecate


deprecate.warn("""
    voxcell.build would be deprecated in voxcell==3.0.
    Please contact NSE team if you are using it.
""")


def sphere_mask(shape, radius):
    '''build the boolean mask of a sphere centered in the middle

    Note that the sphere shape is computed in continuous space but that the returned
    mask has been voxelized, thus breaking the constant radius from the centre.
    As shape and radius become bigger, the voxels cover proportionally less space
    and the resulting shape approaches that of an ideal sphere.

    Args:
        shape: int or sequence of ints. Shape of the new mask.
        radius: float representing the sphere radius in number of voxels
    '''
    mask = np.ones(shape, dtype=np.bool)
    idx = np.nonzero(mask)
    # subtract 1 because python indexes start by 0
    middle = np.floor((np.array(shape) - 1) * 0.5)
    aidx = np.array(idx) - middle[..., np.newaxis]
    mask[idx] = np.sum(np.square(aidx), axis=0) < np.square(radius)
    return mask


def _is_in_triangle(p, v0, v1, v2, epsilon=0.00001):
    '''return True if the point p is inside the triangle defined by the vertices v0, v1, v2'''

    def vector_to(p0, p1):
        '''return a normalized vector p0->p1. Return None if p0 IS p1'''
        to_p1 = p1 - p0
        d2 = np.sum(np.square(to_p1), axis=0)
        if d2 < epsilon:
            return None
        else:
            return to_p1 / np.sqrt(d2)

    to_v0 = vector_to(p, v0)
    to_v1 = vector_to(p, v1)
    to_v2 = vector_to(p, v2)

    if to_v0 is None or to_v1 is None or to_v2 is None:
        return True

    angle = (np.arccos(np.dot(to_v0, to_v1)) +
             np.arccos(np.dot(to_v1, to_v2)) +
             np.arccos(np.dot(to_v2, to_v0)))

    return np.fabs(angle - (2 * np.pi)) < epsilon


def triangular_mask(shape, v0, v1, v2):
    '''build the boolean mask of a 2D triangle

    Args:
        shape(tuple): sequence of two ints. Shape of the new mask.
        v0(numpy.ndarray): 2D vertex of the triangle
        v1(numpy.ndarray): 2D vertex of the triangle
        v2(numpy.ndarray): 2D vertex of the triangle

    Returns:
        A numpy boolean array of the given shape
    '''
    mask = np.ones(shape, dtype=np.bool)
    idx = np.nonzero(mask)
    aidx = np.array(idx).transpose()
    r = np.zeros(aidx.shape[0], dtype=np.bool)  # pylint: disable=unsubscriptable-object

    # TODO make is_in_triangle take arrays of points so we don't need to do one by one
    for i, p in enumerate(aidx):
        r[i] = _is_in_triangle(p, v0, v1, v2)

    mask[idx] = r

    return mask


def regular_convex_polygon_mask(shape, radius, vertex_count):
    '''build the boolean mask of a 2D regular convex polygon
    see https://en.wikipedia.org/wiki/Regular_polygon

    Note that the polygon will be equilateral in continuous space but that the returned
    mask has been voxelized, thus breaking equilaterally with an error range proportional
    to the voxel dimensions. As shape and radius become bigger, the voxels cover
    proportionally less space and the resulting shape approaches equilaterally.
    '''
    assert vertex_count > 2

    angles = np.arange(vertex_count + 1) * ((2 * np.pi) / vertex_count)
    points = radius * np.array([np.cos(angles),
                                np.sin(angles)]).transpose()

    center = (np.array(shape) - 1) * 0.5
    points += center

    mask = np.zeros(shape, dtype=np.bool)
    point_idx = np.arange(vertex_count + 1)

    for i0, i1 in zip(point_idx[:-1], point_idx[1:]):
        m = triangular_mask(shape, points[i0], center, points[i1])
        mask |= m

    return mask


def regular_convex_polygon_mask_from_side(side_size, vertex_count, voxel_size):
    ''' build the boolean mask if a 2D regular convex polygon
    see regular_convex_polygon_mask
    '''
    angle = 2 * np.pi / vertex_count
    radius = (side_size * np.sin((np.pi - angle) / 2.) / np.sin(angle)) / voxel_size
    shape = (int(2 * radius), int(2 * radius))
    return regular_convex_polygon_mask(shape, radius, vertex_count)


def column_mask(pattern, length, axis):
    '''given a 2D patter, repeat it in 3D to build a column along the given axis'''
    column = np.repeat([pattern], repeats=length, axis=0)
    return np.swapaxes(column, 0, axis)


def lattice_tiling(n0, n1, v0, v1, ignore=None):
    '''create a sequence of points representing the origin of the tiles in a pattern

    Args:
        n0: number of elements in the first dimension
        n1: number of elements in the second dimension
        v0: first 2D vector of the lattice base
        v0: second 2D vector of the lattice base
        ignore: a sequence of coordinates to be skipped

    Returns:
        Sequence of numpy arrays representing 2D points
    '''
    ignore = ignore or []

    for i in range(0, n0):
        for j in range(0, n1):
            if (i, j) in ignore:
                continue

            yield ((i + j * 2 - (j // 2)) * v0 +
                   (i * 2 + j - (j // 2) * 2) * v1)


def tiled_pattern(pattern, tiling):
    '''repeat a 2D pattern several times

    Args:
        pattern: 2D boolean numpy array
        tiling: sequence of 2D coordinates for the individual tiles origin
            the values of the coordinates must be integers representing voxel coordinates

    Returns:
        2D boolean numpy array
    '''

    tiling = list(tiling)
    shape = np.array(pattern.shape) + np.max(tiling, axis=0)

    result = np.zeros(shape, dtype=np.bool)

    for origin in tiling:
        result[origin[0]: origin[0] + pattern.shape[0],
               origin[1]: origin[1] + pattern.shape[1]] |= pattern

    return result


def density_from_positions(positions, voxel_dimensions, dtype=np.uint8):
    '''calculate density from the positions'''
    if positions.shape[0] == 0:
        return VoxelData(np.zeros([1] * len(voxel_dimensions), dtype=dtype), voxel_dimensions)

    else:
        aabb_min, aabb_max = math_utils.positions_minimum_aabb(positions)

        dimensions = np.floor((aabb_max - aabb_min) / voxel_dimensions).astype(np.uint)
        dimensions += np.ones_like(dimensions)

        voxels = VoxelData(np.zeros(dimensions, dtype=dtype), voxel_dimensions, offset=aabb_min)

        voxel_indices = voxels.positions_to_indices(positions)

        for x, y, z in voxel_indices:
            # need to iterate because some indices may be repeated
            voxels.raw[x, y, z] += 1

        return voxels


def homogeneous_density(mask, voxel_dimensions, offset=None, value=255):
    '''build an artificial homogeneous density'''
    raw = np.zeros(mask.shape, dtype=np.uint8)
    raw[mask] = value
    return VoxelData(raw, voxel_dimensions=voxel_dimensions, offset=offset)


def layered_annotation(shape, heights, layer_ids):
    ''' build an artificial annotation composed of layers along the Y axis

    Args:
        shape: 2-tuple with the size of the resulting array in X and Z in number of voxels
        heights: sequence of layer heights in number of voxels from lower to higher layer
        layer_ids: sequence of layer ids ordered from lower to higher layer
    '''
    assert len(layer_ids) == len(heights)
    boundaries = np.zeros(len(heights) + 1, dtype=np.uint)
    boundaries[1:] = np.cumsum(heights)

    raw = np.zeros((shape[0], boundaries[-1], shape[1]), dtype=np.uint32)

    idx = 0
    for i, j in zip(boundaries[:-1], boundaries[1:]):
        raw[:, i:j, :] = layer_ids[idx]
        idx += 1

    return raw


def mask_by_region_ids(annotation_raw, region_ids):
    '''get a binary voxel mask where the voxel belonging to the given region ids are True'''

    in_region = np.in1d(annotation_raw, list(region_ids))
    in_region = in_region.reshape(np.shape(annotation_raw))
    return in_region


def mask_by_region_names(annotation_raw, hierarchy, names):
    '''get a binary voxel mask where the voxel belonging to the given region names are True'''
    all_ids = []
    for n in names:
        ids = hierarchy.collect('name', n, 'id')
        if not ids:
            raise KeyError(n)
        all_ids.extend(ids)

    return mask_by_region_ids(annotation_raw, all_ids)


def get_voxel_side(layer_heights, min_value=5):
    ''' compute an optimize size for the voxels '''
    rounded_heights = np.round(layer_heights)
    result = rounded_heights[0]
    for n in rounded_heights:
        result = math_utils.gcd(result, n)
    result = max(min_value, result)
    return result
