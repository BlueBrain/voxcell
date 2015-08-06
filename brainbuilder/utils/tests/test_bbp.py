import os
import numpy as np
from nose.tools import eq_
from numpy.testing import assert_equal

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
    layer_dists = bbp.load_recipe_as_layer_distributions(
        os.path.join(DATA_PATH, 'builderRecipeAllPathways.xml'))

    eq_(layer_dists.keys(), [1, 2])
    eq_([len(l) for l in layer_dists.values()],
        [2, 7])


def test_combine_distributions_empty_0():
    eq_(bbp.combine_distributions([]), {})


def test_combine_distributions_empty_1():
    eq_(bbp.combine_distributions([{}, {}]), {})


def test_combine_distributions_single_0():
    eq_(bbp.combine_distributions([{'a': 0.5, 'b': 0.5}]), {'a': 0.5, 'b': 0.5})


def test_combine_distributions_single_1():
    eq_(bbp.combine_distributions([{'a': 0.5, 'b': 0.5}, {}]), {'a': 0.5, 'b': 0.5})


def test_combine_distributions_2():
    eq_(bbp.combine_distributions([{'a': 0.5, 'b': 0.5}, {'b': 0.5, 'c': 0.5}]),
        {'a': 0.25, 'b': 0.5, 'c': 0.25})


def test_transform_into_spatial_distribution():
    annotation_raw = np.ones(shape=(3, 3, 3))
    annotation_raw[1, 1, 1] = 2

    layer_distributions = {
        21: [(0.5, {'name': 'a'}), (0.5, {'name': 'b'})],
        22: [(0.5, {'name': 'c'}), (0.5, {'name': 'd'})],
        23: [(0.5, {'name': 'e'}), (0.5, {'name': 'f'})]
    }

    region_layers_map = {
        1: (21, 22),
        2: (23,)
    }

    (traits_field, probabilites, traits_collection) = \
        bbp.transform_recipe_into_spatial_distribution(annotation_raw,
                                                       layer_distributions,
                                                       region_layers_map)

    eq_(probabilites,
        [{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
         {4: 0.5, 5: 0.5}])

    eq_(traits_collection,
        [{'name': 'a'}, {'name': 'b'}, {'name': 'c'}, {'name': 'd'}, {'name': 'e'}, {'name': 'f'}])

    expected_field = np.zeros(shape=(3, 3, 3))
    expected_field[1, 1, 1] = 1
    assert_equal(traits_field, expected_field)


def test_load_neurondb_v4():
    morphs = bbp.load_neurondb_v4(os.path.join(DATA_PATH, 'neuronDBv4.dat'))

    eq_(len(morphs), 6)
    for m in morphs:
        eq_(m.keys(), ['layer', 'mtype', 'etype', 'metype', 'morphology', 'placement_hints'])


def test_transform_neurondb_into_spatial_distribution_empty():
    sd = bbp.transform_neurondb_into_spatial_distribution(
        np.ones(shape=(3, 3), dtype=np.int), [], {})

    eq_(sd.traits, [])
    eq_(sd.distributions, [])
    assert_equal(sd.field, np.ones(shape=(3, 3)) * -1)


def test_transform_neurondb_into_spatial_distribution():

    annotation = np.ones(shape=(3, 3), dtype=np.int) * 10
    annotation[1, 1] = 20

    region_layers_map = {
        10: (1,),
        20: (1, 2, 3)
    }

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1,)},
        {'name': 'b', 'layer': 1, 'placement_hints': (1,)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 3, 'placement_hints': (1,)},
    ]

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    eq_(sd.traits, neurondb)

    eq_(sd.distributions,
        [{0: 0.5, 1: 0.5},
         {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}])

    expected_field = np.zeros_like(annotation)
    expected_field[1, 1] = 1
    assert_equal(sd.field, expected_field)


def test_get_region_distributions_from_placement_hints_0():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        0: (1, 2)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map)

    eq_(res.keys(), [0])

    eq_(res[0],
        [{0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
         {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
         {0: 1/6., 1: 2/6., 2: 1/6., 3: 2/6.},
         {0: 2/6., 1: 1/6., 2: 1/6., 3: 2/6.},
         {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2},
         {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}])


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_0():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0] + [1] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    eq_(sd.traits, neurondb)

    eq_(sd.distributions, [
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 1/6.0, 1: 2/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 2/6.0, 1: 1/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
    ])

    assert_equal(sd.field, np.array([-1, 0, 1, 2, 3, 4, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_1():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 12)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    eq_(sd.traits, neurondb)

    eq_(sd.distributions, [
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 1/6.0, 1: 2/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 2/6.0, 1: 1/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
    ])

    assert_equal(sd.field, np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_undivisable():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        1: (1, 2)
    }

    annotation = np.array([0, 0] + [1] * 10)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    eq_(sd.traits, neurondb)

    eq_(sd.distributions, [
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2},
        {0: 1/6.0, 1: 2/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 2/6.0, 1: 1/6.0, 2: 1/6.0, 3: 2/6.0},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2},
        {0: 0.4, 1: 0.2, 2: 0.2, 3: 0.2}
    ])

    assert_equal(sd.field, np.array([-1, -1, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5]))


def test_get_region_distributions_from_placement_hints_multiple_regions():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    res = bbp.get_region_distributions_from_placement_hints(neurondb, region_layers_map)

    eq_(res,
        {1: [{0: 1/3., 1: 2/3.},
             {0: 2/3., 1: 1/3.}],
         2: [{2: 0.5, 3: 0.5},
             {2: 1/3., 3: 2/3.},
             {2: 0.5, 3: 0.5}]})


def test_transform_neurondb_into_spatial_distribution_with_placement_hints_multiple_regions():

    neurondb = [
        {'name': 'a', 'layer': 1, 'placement_hints': (1, 2)},
        {'name': 'b', 'layer': 1, 'placement_hints': (2, 1)},
        {'name': 'c', 'layer': 2, 'placement_hints': (1,)},
        {'name': 'd', 'layer': 2, 'placement_hints': (1, 2, 1)},
    ]

    region_layers_map = {
        1: (1,),
        2: (2,)
    }

    annotation = np.array([0, 0] + [1] * 4 + [2] * 6)

    sd = bbp.transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map)

    eq_(sd.traits, neurondb)

    eq_(sd.distributions, [
        {0: 1/3.0, 1: 2/3.0},
        {0: 2/3.0, 1: 1/3.0},
        {2: 1/2.0, 3: 1/2.0},
        {2: 1/3.0, 3: 2/3.0},
        {2: 1/2.0, 3: 1/2.0}
    ])

    assert_equal(sd.field, np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_assign_distributions_to_voxels_empty_0():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([]), []), np.array([]))


def test_assign_distributions_to_voxels_empty_1():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([]), [{0: 1.}]), np.array([]))


def test_assign_distributions_to_voxels_empty_2():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0]), []), np.array([-1]))


def test_assign_distributions_to_voxels_single():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2]), [{0: 1.}]),
                 np.array([0, 0, 0]))


def test_assign_distributions_to_voxels_more_dists_than_voxels():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0]), [{0: 1.}, {0: 1.}, {0: 1.}]),
                 np.array([1]))


def test_assign_distributions_to_voxels_ascending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([0, 1, 2, 3]), [{}, {}]),
                 np.array([0, 0, 1, 1]))


def test_assign_distributions_to_voxels_descending():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 2, 1, 0]), [{}, {}]),
                 np.array([1, 1, 0, 0]))


def test_assign_distributions_to_voxels_unordered():
    assert_equal(bbp.assign_distributions_to_voxels(np.array([3, 1, 2, 0]), [{}, {}]),
                 np.array([1, 0, 1, 0]))

