import os
from nose.tools import eq_, raises

from voxcell.exceptions import VoxcellError
import voxcell.hierarchy as test_module


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

TEST_RMAP = {
    'id': 1,
    'name': 'A',
    'children': [
        {
            'id': 2,
            'name': 'B',
        },
        {
            'id': 3,
            'name': 'C',
            'children': [
                {
                    'id': 4,
                    'name': 'B',
                }
            ]
        }
    ]
}


def test_from_json():
    filepath = os.path.join(DATA_PATH, 'region_map.json')
    rmap = test_module.RegionMap.from_json(filepath)
    eq_(rmap.get(567), {
        "atlas_id": 70,
        "ontology_id": 1,
        "acronym": "CH",
        "name": "Cerebrum",
        "color_hex_triplet": "B0F0FF",
        "graph_order": 2,
        "st_level": None,
        "hemisphere_id": 3,
        "parent_structure_id": 8
    })


@raises(VoxcellError)
def test_duplicate_id():
    test_module.RegionMap.from_dict({
        'id': 1,
        'children': [
            {
                'id': 2
            },
            {
                'id': 1
            }
        ]
    })


def test_get():
    rmap = test_module.RegionMap.from_dict(TEST_RMAP)
    eq_(rmap.get(1), {'name': 'A'})
    eq_(rmap.get(4), {'name': 'B'})


def test_ascendants():
    rmap = test_module.RegionMap.from_dict(TEST_RMAP)
    eq_(rmap.ascendants(1), [1])
    eq_(rmap.ascendants(3), [3, 1])


def test_descendants():
    rmap = test_module.RegionMap.from_dict(TEST_RMAP)
    eq_(rmap.descendants(1), set([1, 2, 3, 4]))
    eq_(rmap.descendants(3), set([3, 4]))


def test_ids():
    rmap = test_module.RegionMap.from_dict(TEST_RMAP)
    eq_(rmap.ids('B'), set([2, 4]))
    eq_(rmap.ids('C'), set([3, 4]))
    eq_(rmap.ids('C', with_descendants=False), set([3]))

def test_sub():
    rmap = test_module.RegionMap.from_dict(TEST_RMAP)
    sub = rmap.sub(3)
    eq_(sub.ids('A'), set())
    eq_(sub.ids('B'), set([4]))
