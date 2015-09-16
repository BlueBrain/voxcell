import numpy as np
from nose.tools import raises
from numpy.testing import assert_array_equal

from brainbuilder import select_region as sr


def test_sr_0():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.array([[0,  1],
                               [0,  1]], dtype=np.int)

    h = {
        'id': 0,
        'name': 'root',
        'children': [{'id': 1, 'name': 'r', 'children': []}]}

    density_in_region = sr.select_region(annotation_raw, density_raw, h, 'r')

    assert_array_equal(density_in_region,
                       np.array([[0.,  2.],
                                 [0.,  2.]]))


def test_sr_inverse_0():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.array([[0,  1],
                               [0,  1]], dtype=np.int)

    h = {
        'id': 0,
        'name': 'root',
        'children': [{'id': 1, 'name': 'r', 'children': []}]}

    density_in_region = sr.select_region(annotation_raw, density_raw, h, 'r', inverse=True)

    assert_array_equal(density_in_region,
                       np.array([[2.,  0.],
                                 [2.,  0.]]))


@raises(KeyError)
def test_sr_1():
    density_raw = np.ones((2, 2)) * 2
    annotation_raw = np.zeros((2, 2))

    annotation_raw[0, 1] = 1
    annotation_raw[1, 1] = 1

    h = {
        'id': 0,
        'name': 'root',
        'children': [{'id': 1, 'name': 'r', 'children': []}]}

    sr.select_region(annotation_raw, density_raw, h, 'w')
