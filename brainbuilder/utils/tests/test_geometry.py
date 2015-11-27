from brainbuilder.utils import geometry
import numpy as np
from numpy.testing import assert_equal
from nose.tools import eq_


def test_build_sphere_mask_odd_0():
    assert_equal(geometry.build_sphere_mask((3, 3), 0),
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_1():
    assert_equal(geometry.build_sphere_mask((3, 3), 1),
                 np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_2():
    assert_equal(geometry.build_sphere_mask((3, 3), 1.25),
                 np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_3():
    assert_equal(geometry.build_sphere_mask((11, 9), 1),
                 np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_4():
    mask = geometry.build_sphere_mask((1001, 901), 1)
    expected = np.zeros((1001, 901), dtype=np.bool)
    expected[500, 450] = True
    assert_equal(mask, expected)


def test_build_sphere_mask_even_0():
    assert_equal(geometry.build_sphere_mask((4, 4), 0),
                 np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_1():
    assert_equal(geometry.build_sphere_mask((4, 4), 1),
                 np.array([[0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_2():
    assert_equal(geometry.build_sphere_mask((4, 4), 1.25),
                 np.array([[0, 1, 0, 0],
                           [1, 1, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_3():
    mask = geometry.build_sphere_mask((1002, 902), 1)
    expected = np.zeros((1002, 902), dtype=np.bool)
    expected[500, 450] = True
    assert_equal(mask, expected)


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


def test_lattice_tiling_0():
    r = list(geometry.lattice_tiling(0, 0, np.array([1, 0]), np.array([0, 1])))
    eq_(r, [])


def test_lattice_tiling_3():
    r = np.array(list(geometry.lattice_tiling(3, 3, np.array([1, 0]), np.array([0, 1]))))
    assert_equal(r, np.array([[0, 0],
                              [2, 1],
                              [3, 0],
                              [1, 2],
                              [3, 3],
                              [4, 2],
                              [2, 4],
                              [4, 5],
                              [5, 4]]))


def test_build_tiled_pattern_0():
    r = geometry.build_tiled_pattern(np.array([[True]]), [[0, 0], [1, 1]])
    assert_equal(r, np.array([[1, 0],
                              [0, 1]], dtype=np.bool))
