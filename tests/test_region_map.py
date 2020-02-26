import json

from nose.tools import eq_, raises
from mock import mock_open, patch

from voxcell.exceptions import VoxcellError
import voxcell.region_map as test_module


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
    eq_(TEST_RMAP.find('B', 'name'), set([2, 4]))
    eq_(TEST_RMAP.find('C', 'name'), set([3]))
    eq_(TEST_RMAP.find('cC', attr='fullname'), set([3]))


def test_find_regex():
    eq_(TEST_RMAP.find('B', attr='fullname'), set())
    eq_(TEST_RMAP.find('@B', attr='fullname'), set([2, 4]))
    eq_(TEST_RMAP.find('@^B', attr='fullname'), set([2]))
    eq_(TEST_RMAP.find('@B$', attr='fullname'), set([4]))


def test_find_with_descendants():
    eq_(TEST_RMAP.find('B', 'name', with_descendants=True), set([2, 4]))
    eq_(TEST_RMAP.find('C', 'name', with_descendants=True), set([3, 4]))


def test_find_ignore_case():
    eq_(TEST_RMAP.find('cc', attr='fullname'), set())
    eq_(TEST_RMAP.find('cc', attr='fullname', ignore_case=True), set([3]))
    eq_(TEST_RMAP.find('CC', attr='fullname', ignore_case=True), set([3]))

def test_find_regex_ignore_case():
    eq_(TEST_RMAP.find('@^B', attr='fullname'), set([2]))
    eq_(TEST_RMAP.find('@^B', attr='fullname', ignore_case=True), set([2, 4]))


def test_find_id():
    eq_(TEST_RMAP.find(1, attr='id', with_descendants=True), set([1, 2, 3, 4]))
    eq_(TEST_RMAP.find(1, attr='id', ignore_case=True), set([1]))
    eq_(TEST_RMAP.find(999, attr='id'), set())


@raises(VoxcellError)
def test_find_missing_attribute():
    TEST_RMAP.find(1, attr='no-such-attribute')


def test_get_basic():
    eq_(TEST_RMAP.get(4, 'name'), 'B')
    eq_(TEST_RMAP.get(4, attr='fullname'), 'bB')


def test_get_with_ascendants():
    eq_(TEST_RMAP.get(4, 'name', with_ascendants=True), ['B', 'C', 'A'])


@raises(VoxcellError)
def test_get_missing_attribute():
    TEST_RMAP.get(4, 'no-such-attribute')


@raises(VoxcellError)
def test_get_missing_id():
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
    eq_(rmap.get(42, 'name'), "foo")


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
    eq_(rmap.get(42, 'name'), "foo")


@raises(VoxcellError)
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
    with _patch_open_json(data):
        test_module.RegionMap.load_json('mock-file')


@raises(VoxcellError)
def test_from_dict_duplicate_id():
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
    eq_(TEST_RMAP.is_leaf_id(1), False)
    eq_(TEST_RMAP.is_leaf_id(2), True)
    eq_(TEST_RMAP.is_leaf_id(3), False)
    eq_(TEST_RMAP.is_leaf_id(4), True)


@raises(VoxcellError)
def test_is_leaf_id_non_existing_id():
    TEST_RMAP.is_leaf_id(0)  # non-existing id
