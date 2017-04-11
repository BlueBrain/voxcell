from nose.tools import eq_

import numpy as np
from numpy.testing import assert_equal
from voxcell import build


def test_cell_density_from_positions_0():
    positions = np.zeros((0, 3))
    voxel_dimensions = (25, 25, 25)

    result = build.density_from_positions(positions, voxel_dimensions)

    assert_equal(result.raw, np.zeros((1, 1, 1)))
    assert_equal(result.offset, np.array([0, 0, 0]))


def test_cell_density_from_positions_1():
    positions = np.zeros((1, 3))
    voxel_dimensions = (25, 25, 25)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.ones((1, 1, 1))
    assert_equal(result.raw, expected)
    assert_equal(result.offset, np.array([0, 0, 0]))


def test_cell_density_from_positions_random_cube():
    positions = np.random.random((10, 3))
    voxel_dimensions = (25, 25, 25)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.ones((1, 1, 1)) * 10
    assert_equal(result.raw, expected)


def test_cell_density_from_positions_negative_0():
    positions = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    voxel_dimensions = (25, 25, 25)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.ones((1, 1, 1)) * 3
    assert_equal(result.raw, expected)
    assert_equal(result.offset, np.array([-1, -1, -1]))


def test_cell_density_from_positions_negative_1():
    positions = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    voxel_dimensions = (1, 1, 1)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.zeros((3, 3, 3))
    expected[0, 0, 0] = 1
    expected[1, 1, 1] = 1
    expected[2, 2, 2] = 1
    assert_equal(result.raw, expected)
    assert_equal(result.offset, np.array([-1, -1, -1]))


def test_cell_density_from_positions_negative_2():
    positions = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    voxel_dimensions = (0.5, 0.5, 0.5)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.zeros((5, 5, 5))
    expected[0, 0, 0] = 1
    expected[2, 2, 2] = 1
    expected[4, 4, 4] = 1
    assert_equal(result.raw, expected)
    assert_equal(result.offset, np.array([-1, -1, -1]))


def test_cell_density_from_positions_negative_3():
    positions = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    voxel_dimensions = (0.5, 1, 25)

    result = build.density_from_positions(positions, voxel_dimensions)

    expected = np.zeros((5, 3, 1))
    expected[0, 0, 0] = 1
    expected[2, 1, 0] = 1
    expected[4, 2, 0] = 1
    assert_equal(result.raw, expected)
    assert_equal(result.offset, np.array([-1, -1, -1]))


def test_cell_density_from_positions_homogeneous():
    density_dimensions = (3, 4, 5)

    positions = np.zeros((np.prod(density_dimensions), 3))
    idx = 0
    for i in range(density_dimensions[0]):
        for j in range(density_dimensions[1]):
            for k in range(density_dimensions[2]):
                positions[idx] = [i, j, k]
                idx += 1

    voxel_dimensions = (1, 1, 1)

    result = build.density_from_positions(positions, voxel_dimensions)

    assert_equal(result.raw, np.ones(density_dimensions))
    assert_equal(result.offset, np.array([0, 0, 0]))


def test_cell_density_from_positions_homogeneous_negative():
    density_dimensions = (3, 4, 5)

    positions = np.zeros((np.prod(density_dimensions), 3))
    idx = 0
    for i in range(density_dimensions[0]):
        for j in range(density_dimensions[1]):
            for k in range(density_dimensions[2]):
                positions[idx] = [-i, -j, -k]
                idx += 1

    voxel_dimensions = (1, 1, 1)

    result = build.density_from_positions(positions, voxel_dimensions)

    assert_equal(result.raw, np.ones(density_dimensions))
    assert_equal(result.offset, np.array([-2, -3, -4]))


def test_build_homogeneous_density_empty():
    assert_equal(build.homogeneous_density(np.array([], dtype=np.bool), (1,)).raw,
                 np.array([]))


def test_build_homogeneous_density_1():
    assert_equal(build.homogeneous_density(np.array([0, 1, 0], dtype=np.bool), (1,)).raw,
                 np.array([0, 255, 0]))


def test_build_layered_annotation_empty_0():
    assert_equal(build.layered_annotation((10, 10), [], []),
                 np.empty((10, 0, 10)))


def test_build_layered_annotation_empty_1():
    assert_equal(build.layered_annotation((10, 10), [0], [1]),
                 np.empty((10, 0, 10)))


def test_build_layered_annotation_homogeneous():
    assert_equal(build.layered_annotation((10, 10), [10], [1]),
                 np.ones((10, 10, 10)))


def test_build_layered_annotation_same_heights():
    assert_equal(build.layered_annotation((1, 1), [1] * 5, range(5)),
                 np.array([[[0], [1], [2], [3], [4]]]))


def test_build_layered_annotation_different_heights():
    assert_equal(build.layered_annotation((1, 1), range(4), range(4)),
                 np.array([[[1], [2], [2], [3], [3], [3]]]))


def test_mask_by_region_ids():
    assert_equal(build.mask_by_region_ids(np.ones((3, 3)), set([1])),
                 np.ones((3, 3), dtype=np.bool))

    assert_equal(build.mask_by_region_ids(np.ones((3, 3)), set([3])),
                 np.zeros((3, 3), dtype=np.bool))


def test_build_sphere_mask_odd_0():
    assert_equal(build.sphere_mask((3, 3), 0),
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_1():
    assert_equal(build.sphere_mask((3, 3), 1),
                 np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_2():
    assert_equal(build.sphere_mask((3, 3), 1.25),
                 np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_3():
    assert_equal(build.sphere_mask((11, 9), 1),
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
    mask = build.sphere_mask((1001, 901), 1)
    expected = np.zeros((1001, 901), dtype=np.bool)
    expected[500, 450] = True
    assert_equal(mask, expected)


def test_build_sphere_mask_even_0():
    assert_equal(build.sphere_mask((4, 4), 0),
                 np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_1():
    assert_equal(build.sphere_mask((4, 4), 1),
                 np.array([[0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_2():
    assert_equal(build.sphere_mask((4, 4), 1.25),
                 np.array([[0, 1, 0, 0],
                           [1, 1, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_3():
    mask = build.sphere_mask((1002, 902), 1)
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
        assert build._is_in_triangle(v0, v0, v1, v2)
        assert build._is_in_triangle(v1, v0, v1, v2)
        assert build._is_in_triangle(v2, v0, v1, v2)
        assert build._is_in_triangle(np.mean([v0, v1, v2], axis=0), v0, v1, v2)


def test_build_2d_triangular_mask_0():
    mask = build.triangular_mask((3, 3),
                                 np.array([0, 0]), np.array([1, 2]), np.array([2, 0]))

    assert_equal(mask, np.array([[1, 0, 0],
                                 [1, 1, 1],
                                 [1, 0, 0]], dtype=np.bool))


def test_build_2d_regular_convex_polygon_mask():
    mask = build.regular_convex_polygon_mask((3, 3), 0.5, 10)

    assert_equal(mask, np.array([[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]], dtype=np.bool))


def test_build_2d_regular_convex_polygon_mask_from_side():
    mask = build.regular_convex_polygon_mask_from_side(20, 6, 10)

    assert_equal(mask, np.array([[0, 1, 1, 0],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 1, 0]], dtype=np.bool))


def test_hexagon_corners():
    hexagon = build.regular_convex_polygon_mask((11, 11), 5, 6)

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

    assert_equal(build.column_mask(pattern, length=2, axis=0),
                 np.array([[[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]],
                           [[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]]))

    eq_(build.column_mask(pattern, length=2, axis=0).shape, (2, 3, 3))
    eq_(build.column_mask(pattern, length=2, axis=1).shape, (3, 2, 3))
    eq_(build.column_mask(pattern, length=2, axis=2).shape, (3, 3, 2))


def test_lattice_tiling_0():
    r = list(build.lattice_tiling(0, 0, np.array([1, 0]), np.array([0, 1])))
    eq_(r, [])


def test_lattice_tiling_3():
    r = np.array(list(build.lattice_tiling(3, 3, np.array([1, 0]), np.array([0, 1]))))
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
    r = build.tiled_pattern(np.array([[True]]), [[0, 0], [1, 1]])
    assert_equal(r, np.array([[1, 0],
                              [0, 1]], dtype=np.bool))
