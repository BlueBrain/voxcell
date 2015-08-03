import os
import numpy as np
from nose.tools import eq_
from numpy.testing import assert_equal

from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import bbp


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_map_regions_to_layers_unknown_area():
    regions_layers_map = bbp.map_regions_to_layers({"id": 997, "name": "mybrain", "children": []},
                                                   'Primary somatosensory area, barrel field')

    eq_(regions_layers_map, {})


def test_map_regions_to_layers_complex():
    hierarchy = gb.load_hierarchy(os.path.join(DATA_PATH, 'hierarchy.json'))

    regions_layers_map = bbp.map_regions_to_layers(hierarchy,
                                                   'Primary somatosensory area, barrel field')

    eq_(regions_layers_map,
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
        bbp.transform_into_spatial_distribution(annotation_raw,
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
