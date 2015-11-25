'''helper geometrical functions'''

import numpy as np


def build_sphere_mask(shape, radius):
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


def is_in_triangle(p, v0, v1, v2, epsilon=0.00001):
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


def build_2d_triangular_mask(shape, v0, v1, v2):
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
    r = np.zeros(aidx.shape[0], dtype=np.bool)

    # TODO make is_in_triangle take arrays of points so we don't need to do one by one
    for i, p in enumerate(aidx):
        r[i] = is_in_triangle(p, v0, v1, v2)

    mask[idx] = r

    return mask


def build_2d_regular_convex_polygon_mask(shape, radius, vertex_count):
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
        m = build_2d_triangular_mask(shape, points[i0], center, points[i1])
        mask |= m

    return mask


def build_column_mask(pattern, length, axis):
    '''given a 2D patter, repeat it in 3D to build a column along the given axis'''
    column = np.repeat([pattern], repeats=length, axis=0)
    return np.swapaxes(column, 0, axis)
