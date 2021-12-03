import json
from unittest.mock import mock_open, patch

import pytest

import voxcell.region_map as test_module
from voxcell.exceptions import VoxcellError

TEST_RMAP = test_module.RegionMap.from_dict({
    'id': 1,
    'name': 'A',
    'fullname': 'aA',
    'children': [
        {
            'id': 2,
            'name': 'B',
            'fullname': 'Bb',
        },
        {
            'id': 3,
            'name': 'C',
            'fullname': 'cC',
            'children': [
                {
                    'id': 4,
                    'name': 'B',
                    'fullname': 'bB',
                }
            ]
        }
    ]
})


def test_find_basic():
    assert TEST_RMAP.find('B', 'name') == {2, 4}
    assert TEST_RMAP.find('C', 'name') == {3}
    assert TEST_RMAP.find('cC', attr='fullname') == {3}


def test_find_regex():
    assert TEST_RMAP.find('B', attr='fullname') == set()
    assert TEST_RMAP.find('@B', attr='fullname') == {2, 4}
    assert TEST_RMAP.find('@^B', attr='fullname') == {2}
    assert TEST_RMAP.find('@B$', attr='fullname') == {4}


def test_find_with_descendants():
    assert TEST_RMAP.find('B', 'name', with_descendants=True) == {2, 4}
    assert TEST_RMAP.find('C', 'name', with_descendants=True) == {3, 4}


def test_find_ignore_case():
    assert TEST_RMAP.find('cc', attr='fullname') == set()
    assert TEST_RMAP.find('cc', attr='fullname', ignore_case=True) == {3}
    assert TEST_RMAP.find('CC', attr='fullname', ignore_case=True) == {3}

def test_find_regex_ignore_case():
    assert TEST_RMAP.find('@^B', attr='fullname') == {2}
    assert TEST_RMAP.find('@^B', attr='fullname', ignore_case=True) == {2, 4}


def test_find_id():
    assert TEST_RMAP.find(1, attr='id', with_descendants=True) == {1, 2, 3, 4}
    assert TEST_RMAP.find(1, attr='id', ignore_case=True) == {1}
    assert TEST_RMAP.find(999, attr='id') == set()


def test_find_missing_attribute():
    with pytest.raises(VoxcellError):
        TEST_RMAP.find(1, attr='no-such-attribute')


def test_get_basic():
    assert TEST_RMAP.get(4, 'name') == 'B'
    assert TEST_RMAP.get(4, attr='fullname') == 'bB'


def test_get_with_ascendants():
    assert TEST_RMAP.get(4, 'name', with_ascendants=True) == ['B', 'C', 'A']


def test_get_missing_attribute():
    with pytest.raises(VoxcellError):
        TEST_RMAP.get(4, 'no-such-attribute')


def test_get_missing_id():
    with pytest.raises(VoxcellError):
        TEST_RMAP.get(999, 'name')


def _patch_open_json(data):
    return patch('voxcell.region_map.open', new_callable=mock_open, read_data=json.dumps(data))


def test_load_json_basic():
    data = {
        'id': 42,
        'name': "foo",
    }
    with _patch_open_json(data):
        rmap = test_module.RegionMap.load_json('mock-file')
    assert rmap.get(42, 'name') == "foo"


def test_load_json_aibs():
    data = {
        'msg': [
            {
                'id': 42,
                'name': "foo",
            },
        ]
    }
    with _patch_open_json(data):
        rmap = test_module.RegionMap.load_json('mock-file')
    assert rmap.get(42, 'name') == "foo"


def test_load_json_aibs_raises():
    data = {
        'msg': [
            {
                'id': 42,
                'name': "foo",
            },
            {
                'id': 43,
                'name': "bar",
            }
        ]
    }
    with _patch_open_json(data), pytest.raises(VoxcellError):
        test_module.RegionMap.load_json('mock-file')


def test_from_dict_duplicate_id():
    with pytest.raises(VoxcellError):
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


def test_is_leaf_id():
    assert TEST_RMAP.is_leaf_id(1) is False
    assert TEST_RMAP.is_leaf_id(2) is True
    assert TEST_RMAP.is_leaf_id(3) is False
    assert TEST_RMAP.is_leaf_id(4) is True


def test_is_leaf_id_non_existing_id():
    with pytest.raises(VoxcellError):
        TEST_RMAP.is_leaf_id(0)  # non-existing id
