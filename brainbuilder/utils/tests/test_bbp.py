import os
import numpy as np
import pandas as pd
from nose.tools import eq_
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal

from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import bbp


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_map_regions_to_layers_unknown_area():
    region_layers_map = bbp.map_regions_to_layers({"id": 997, "name": "mybrain", "children": []},
                                                   'Primary somatosensory area, barrel field')

    eq_(region_layers_map, {})


def test_map_regions_to_layers_complex():
    hierarchy = gb.load_hierarchy(os.path.join(DATA_PATH, 'hierarchy.json'))

    region_layers_map = bbp.map_regions_to_layers(hierarchy,
                                                   'Primary somatosensory area, barrel field')

    eq_(region_layers_map,
        {1062: (6,), 201: (2, 3), 1070: (5,), 1038: (6,), 981: (1,), 1047: (4,)})


def test_load_recipe_as_layer_distributions_complex():
    layer_dists = bbp.load_recipe(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'))

    eq_(list(layer_dists.keys()),
                 ['layer', 'mtype', 'etype', 'mClass', 'sClass', 'percentage'])

    eq_(len(layer_dists.values), 9)


def test_transform_into_spatial_distribution():
    annotation_raw = np.ones(shape=(3, 3, 3))
    annotation_raw[1, 1, 1] = 2

    layer_distributions = pd.DataFrame([
        {'layer': 21, 'percentage': 0.5},
        {'layer': 22, 'percentage': 0.5},
        {'layer': 23, 'percentage': 0.5}
    ])

    region_layers_map = {
        1: (21, 22),
        2: (23,)
    }

    (traits_field, probabilites, traits_collection) = \
        bbp.transform_recipe_into_spatial_distribution(annotation_raw,
                                                       layer_distributions,
                                                       region_layers_map)

    assert_frame_equal(probabilites,
                       pd.DataFrame({1: [0.5,  0.5, 0.],
                                     2: [0., 0., 1.]}))

    assert_frame_equal(traits_collection, layer_distributions)

    expected_field = np.ones(shape=(3, 3, 3))
    expected_field[1, 1, 1] = 2
    assert_equal(traits_field, expected_field)


def test_load_neurondb_v4():
    morphs = bbp.load_neurondb_v4(os.path.join(DATA_PATH, 'neuronDBv4.dat'))
    eq_(len(morphs), 6)


def test_transform_neurondb_into_spatial_distribution_empty():
    sd = bbp.transform_neurondb_into_spatial_distribution(
        np.ones(shape=(3, 3), dtype=np.int), pd.DataFrame(), {})

    assert sd.traits.empty
    assert sd.distributions.empty
    assert_equal(sd.field, np.ones(shape=(3, 3)) * -1)


def test_transform_neurondb_into_spatial_distribution():

    annotation = np.ones(shape=(3, 3), dtype=np.int) * 10
    annotation[1, 1] = 20

    region_layers_map = {
        10: (1,),
        20: (1, 2, 3)
    }

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1,)},
        {'name': 'b', 'layer': 1, 'placement_hints': (1,)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 3, 'placement_hints': (1,)},
    ])

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    assert_frame_equal(sd.traits, neurondb)

    print sd.distributions.values

    expected = pd.DataFrame({0: [0.5, 0.5, 0, 0],
                             1: [0.25, 0.25, 0.25, 0.25]})

    print expected.values
    assert_frame_equal(sd.distributions, expected)

    expected_field = np.zeros_like(annotation)
    expected_field[1, 1] = 1
    assert_equal(sd.field, expected_field)


def test_get_region_distributions_from_placement_hints_0():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        0: (1, 2)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map)

    eq_(res.keys(), [(0,)])

    expected = pd.DataFrame({
         0: [1., 2., 1., 1.],
         1: [1., 2., 1., 1.],
         2: [1., 2., 1., 2.],
         3: [2., 1., 1., 2.],
         4: [2., 1., 1., 1.],
         5: [2., 1., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(res[(0,)], expected)


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_0():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0] + [1] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
         0: [1., 2., 1., 1.],
         1: [1., 2., 1., 1.],
         2: [1., 2., 1., 2.],
         3: [2., 1., 1., 2.],
         4: [2., 1., 1., 1.],
         5: [2., 1., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field, np.array([-1, 0, 1, 2, 3, 4, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_1():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 12)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
         0: [1., 2., 1., 1.],
         1: [1., 2., 1., 1.],
         2: [1., 2., 1., 2.],
         3: [2., 1., 1., 2.],
         4: [2., 1., 1., 1.],
         5: [2., 1., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field, np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_undivisable():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 1 1 2 2 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 2 2 1 1 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 1 2 2 1 1
    ])

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 10)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    assert_frame_equal(sd.traits, neurondb)

    expected = pd.DataFrame({
         0: [1., 2., 1., 1.],
         1: [1., 2., 1., 1.],
         2: [1., 2., 1., 2.],
         3: [2., 1., 1., 2.],
         4: [2., 1., 1., 1.],
         5: [2., 1., 1., 1.],
    })
    expected /= expected.sum()
    assert_frame_equal(sd.distributions, expected)

    assert_equal(sd.field, np.array([-1, -1, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5]))


def test_get_region_distributions_from_placement_hints_multiple_regions():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 2 1
    ])

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map)

    eq_(res.keys(), [(2,), (1,)])
    assert_frame_equal(res[(1,)], pd.DataFrame({0: [1/3., 2/3.],
                                                1: [2/3., 1/3.]}))

    assert_frame_equal(res[(2,)], pd.DataFrame({0: [1/2., 1/2.],
                                                1: [1/3., 2/3.],
                                                2: [1/2., 1/2.]}, index=[2, 3]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_multiple_regions():

    neurondb = pd.DataFrame([
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},     # 1 2
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},     # 2 1
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},       # 1 1 1
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},  # 1 2 1
    ])

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    annotation = np.array([0, 0] + [1] * 4 + [2] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    assert_frame_equal(sd.traits, neurondb)

    print sd.distributions

    assert_frame_equal(sd.distributions, pd.DataFrame({
        0: [0.,     0.,  1/2., 1/2.],
        1: [0.,     0.,  1/3., 2/3.],
        2: [0.,     0.,  1/2., 1/2.],
        3: [1/3., 2/3.,    0.,   0.],
        4: [2/3., 1/3.,    0.,   0.]
    }))

    assert_equal(sd.field, np.array([-1, -1, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2]))


def test_assign_distributions_to_voxels_empty():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([]), 1),
                 np.array([]))


def test_assign_distributions_to_voxels_single():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2]), 1),
                 np.array([0, 0, 0]))


def test_assign_distributions_to_voxels_more_dists_than_voxels():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0]), 3),
                 np.array([1]))


def test_assign_distributions_to_voxels_ascending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2, 3]), 2),
                 np.array([0, 0, 1, 1]))


def test_assign_distributions_to_voxels_descending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 2, 1, 0]), 2),
                 np.array([1, 1, 0, 0]))


def test_assign_distributions_to_voxels_unordered():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 1, 2, 0]), 2),
                 np.array([1, 0, 1, 0]))

