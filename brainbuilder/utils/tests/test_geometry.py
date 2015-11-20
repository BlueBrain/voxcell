from brainbuilder.utils import geometry
import numpy as np
from numpy.testing import assert_equal
from nose.tools import eq_


def test_is_in_triangle():
    # points hand picked from a previous hexagon run
    triangles = [(np.array([2.5, 9.33012702]), np.array([5., 5.]), np.array([0., 5.])),
                 (np.array([10., 5., 0.]), np.array([5., 5., 0.]), np.array([7.5, 9.33012702, 0.])),
                 (np.array([0., 5., 0.]), np.array([5., 5., 0.]), np.array([2.5, 0.66987298, 0.]))]

    # test the corners are part of the triangle and that average is too
    for v0, v1, v2 in triangles:
        assert geometry.is_in_triangle(v0, v0, v1, v2)
        assert geometry.is_in_triangle(v1, v0, v1, v2)
        assert geometry.is_in_triangle(v2, v0, v1, v2)
        assert geometry.is_in_triangle(np.mean([v0, v1, v2], axis=0), v0, v1, v2)


def test_build_2d_triangular_mask_0():
    mask = geometry.build_2d_triangular_mask((3, 3),
                                             np.array([0, 0]), np.array([1, 2]), np.array([2, 0]))

    assert_equal(mask, np.array([[1, 0, 0],
                                 [1, 1, 1],
                                 [1, 0, 0]], dtype=np.bool))


def test_build_2d_regular_convex_polygon_mask():
    mask = geometry.build_2d_regular_convex_polygon_mask((3, 3), 0.5, 10)

    assert_equal(mask, np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]], dtype=np.bool))


def test_hexagon_corners():
    hexagon = geometry.build_2d_regular_convex_polygon_mask((11, 11), 5, 6)

    assert hexagon[0, 5]

    # pro tip: squint your eyes
    expected = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

    assert_equal(hexagon, expected)


def test_build_column_mask():
    pattern = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])

    assert_equal(geometry.build_column_mask(pattern, length=2, axis=0),
                 np.array([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]]))

    eq_(geometry.build_column_mask(pattern, length=2, axis=0).shape, (2, 3, 3))
    eq_(geometry.build_column_mask(pattern, length=2, axis=1).shape, (3, 2, 3))
    eq_(geometry.build_column_mask(pattern, length=2, axis=2).shape, (3, 3, 2))
