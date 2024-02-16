import json
from unittest.mock import mock_open, patch

import pytest

import voxcell.region_map as test_module
from voxcell.exceptions import VoxcellError

TEST_RMAP = test_module.RegionMap.from_dict({
    'id': -1,
    'name': 'root',
    'fullname': 'The Root Node',
    'children': [
        {
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
        ]}
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


def test_find_missing_parents():
    with pytest.raises(VoxcellError, match='ID 666 is unknown in the hierarchy'):
        TEST_RMAP.get(666, attr='fullname', with_ascendants=True)


def test_get_basic():
    assert TEST_RMAP.get(4, 'name') == 'B'
    assert TEST_RMAP.get(4, attr='fullname') == 'bB'


def test_get_with_ascendants():
    assert TEST_RMAP.get(4, 'name', with_ascendants=True) == ['B', 'C', 'A', 'root']


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


def test_as_dataframe():
    df = TEST_RMAP.as_dataframe()
    assert df.loc[-1].parent_id == -1
    assert df.loc[1].parent_id == -1
    assert df.loc[2].parent_id == 1
    assert df.loc[3].parent_id == 1
    assert df.loc[4].parent_id == 3

    assert df.loc[1]['name'] == 'A'
    assert df.loc[1]['fullname'] == 'aA'

    assert df.loc[-1].children_count == 1
    assert df.loc[1].children_count == 2
    assert df.loc[2].children_count == 0
    assert df.loc[3].children_count == 1
    assert df.loc[4].children_count == 0


@pytest.mark.parametrize(
    "attr, values, expected",
    [
        (
            "id",
            [(4, "red"), (2, "red")],
            [(-1, "red")],
        ),
        (
            "id",
            [(4, "blue"), (2, "red")],
            [(2, "red"), (3, "blue")],
        ),
        (
            "fullname",
            [("bB", "red"), ("Bb", "red")],
            [("The Root Node", "red")],
        ),
    ],
)
def test_get_common_node_groups_simple(attr, values, expected):
    res = TEST_RMAP.get_common_node_groups(attr=attr, values=values)
    assert res == expected


def test_get_common_node_groups_raises():
    with pytest.raises(VoxcellError, match="Cannot overwrite existing key 1"):
        TEST_RMAP.get_common_node_groups(
            attr="id",
            values=[(1, "green"), (4, "red"), (2, "red")],
        )

    with pytest.raises(VoxcellError, match="Value not found: does-not-exist == green"):
        TEST_RMAP.get_common_node_groups(
            attr="name",
            values=[("does-not-exist", "green")],
        )

    with pytest.raises(VoxcellError, match="Multiple values found for: B == green"):
        TEST_RMAP.get_common_node_groups(
            attr="name",
            values=[("B", "green")],
        )


@pytest.mark.parametrize(
    "attr, values, expected",
    [
        (
            "id",
            [],
            [],
        ),
        (
            "id",
            [(-1, "blue")],
            [(-1, "blue")],
        ),
        (
            "id",
            [(110, "blue")],
            [(-1, "blue")],
        ),
        (
            "id",
            [(11, "blue"), (10, "red")],
            [(10, "red"), (11, "blue")],
        ),
        (
            "id",
            [(110, "blue"), (111, "blue"), (113, "blue"), (10, "red")],
            [(10, "red"), (11, "blue")],
        ),
        (
            "id",
            [(110, "blue"), (111, "blue"), (113, "blue"), (10, "blue")],
            [(-1, "blue")],
        ),
        (
            "id",
            [
                (110, "blue"),
                (111, "blue"),
                (113, "blue"),
                (100, "blue"),
                (101, "blue"),
                (103, "blue"),
            ],
            [(-1, "blue")],
        ),
        (
            "id",
            [
                (100, "blue"),
                (101, "blue"),
                (110, "red"),
                (111, "blue"),
                (113, "blue"),
                (12, "green"),
            ],
            [(10, "blue"), (12, "green"), (110, "red"), (111, "blue"), (113, "blue")],
        ),
    ],
)
def test_get_common_node_groups_complex(attr, values, expected):
    rm = test_module.RegionMap.from_dict(
        {
            "id": -1,
            "children": [
                {
                    "id": 10,
                    "children": [
                        {"id": 100},
                        {"id": 101},
                        {"id": 103},
                    ],
                },
                {
                    "id": 11,
                    "children": [
                        {"id": 110},
                        {"id": 111},
                        {"id": 113},
                    ],
                },
                {
                    "id": 12,
                },
            ],
        }
    )
    res = rm.get_common_node_groups(attr=attr, values=values)
    assert sorted(res) == sorted(expected)
