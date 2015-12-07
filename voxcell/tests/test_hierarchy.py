import os
import tempfile
from nose.tools import eq_, raises

from voxcell import core

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_load_hierarchy_0():
    with tempfile.NamedTemporaryFile() as f:
        f.write('{"msg": [{}]}')
        f.flush()

        h = core.Hierarchy.load(f.name)
        eq_(h.data, {})
        eq_(h.children, [])


def test_load_hierarchy_1():
    h = core.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json'))
    eq_(h.data['name'], 'root')


@raises(KeyError)
def test_find_in_hierarchy_0():
    eq_(core.Hierarchy({}).find('id', 'xxx'), [])


def test_find_in_hierarchy_1():
    eq_(core.Hierarchy({"id": 997, "children": []}).find('id', 'xxx'), [])


def test_find_in_hierarchy_beyond_me():
    found = core.Hierarchy({"id": 997, "prop": "a", "children": []}).find('prop', 'a')
    eq_(len(found), 1)

    found = core.Hierarchy(
        {"id": 997, "prop": "a", "children": [
            {"id": 998, "prop": "a", "children": []}
        ]}).find('prop', 'a')

    eq_(len(found), 2)


def test_find_in_hierarchy_2():
    res = core.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json')).find(
                               'name', 'Primary somatosensory area, barrel field')
    eq_(len(res), 1)
    eq_(res[0].data['name'], 'Primary somatosensory area, barrel field')


@raises(KeyError)
def test_get_in_hierarchy_0():
    core.Hierarchy({}).get('id')


def test_get_in_hierarchy_1():
    eq_(core.Hierarchy({"id": 997, "children": []}).get('id'), set([997]))


def test_get_in_hierarchy_2():
    h = core.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json')).find(
                             'name', 'Primary somatosensory area, barrel field')
    res = h[0].get('id')
    eq_(res, set([329, 981, 201, 1047, 1070, 1038, 1062]))


def test_collect_in_hierarchy():
    h = core.Hierarchy(
        {"id": 997, "prop": "a", "prop2": "a", "children": [
            {"id": 998, "prop": "a", "prop2": "b", "children": []}
        ]})

    eq_(h.collect('prop', 'a', 'prop2'), set(['a', 'b']))


def test_print_hierarchy_empty():
    h = core.Hierarchy({})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = core.Hierarchy({'children': []})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = core.Hierarchy({'children': [{}]})
    eq_(str(h), '<unnamed section>\n'
                '    children: [\n'
                '    <unnamed section>\n'
                '        children: []\n'
                '    ]')


def test_print_hierarchy_props():
    h = core.Hierarchy({'name': 'brains', 'prop': 'a'})
    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: []')

    h = core.Hierarchy({'name': 'brains', 'prop': 'a',
                        'children': [{'name': 'grey stuff', 'prop': 'b'}]})

    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: [\n'
                '    grey stuff\n'
                '        prop: b\n'
                '        children: []\n'
                '    ]')
