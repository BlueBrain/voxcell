from nose.tools import raises

import numpy as np
from numpy.testing import assert_array_equal
from voxcell import Hierarchy, VoxelData

import brainbuilder.select_region as test_module

HIERARCHY = Hierarchy({
    'id': 0,
    'name': 'root',
    'children': [{'id': 1, 'name': 'r', 'children': []}]
})


def test_sr_0():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.array([[0,  1],
                               [0,  1]], dtype=np.int)

    density_in_region = test_module.select_region(annotation_raw, VoxelData(density_raw, (25, 25)),
                                         HIERARCHY, 'r')

    assert_array_equal(density_in_region.raw,
                       np.array([[0.,  2.],
                                 [0.,  2.]]))


def test_sr_inverse_0():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.array([[0,  1],
                               [0,  1]], dtype=np.int)

    density_in_region = test_module.select_region(annotation_raw, VoxelData(density_raw, (25, 25)),
                                         HIERARCHY, 'r', inverse=True)

    assert_array_equal(density_in_region.raw,
                       np.array([[2.,  0.],
                                 [2.,  0.]]))


@raises(KeyError)
def test_sr_1():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.zeros((2, 2))

    annotation_raw[0, 1] = 1
    annotation_raw[1, 1] = 1

    test_module.select_region(annotation_raw, density_raw, HIERARCHY, 'w')


def test_select_hemisphere_default():
    density_raw = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )
    assert_array_equal(
        test_module.select_hemisphere(density_raw),
        [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]
    )


def test_select_hemisphere_default_right():
    density_raw = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )
    assert_array_equal(
        test_module.select_hemisphere(density_raw, left=False),
        [[[1, 2], [3, 4]], [[0, 0], [0, 0]]]
    )


def test_select_hemisphere_PIR():
    density_raw = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )
    assert_array_equal(
        test_module.select_hemisphere(density_raw, orientation="PIR"),
        [[[1, 0], [3, 0]], [[5, 0], [7, 0]]]
    )


def test_select_hemisphere_PIR_right():
    density_raw = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )
    assert_array_equal(
        test_module.select_hemisphere(density_raw, left=False, orientation="PIR"),
        [[[0, 2], [0, 4]], [[0, 6], [0, 8]]]
    )


@raises(ValueError)
def test_select_hemisphere_raises():
    density_raw = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    )
    test_module.select_hemisphere(density_raw, orientation="err")