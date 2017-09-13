import os
import tempfile
from nose.tools import eq_, raises

import voxcell.hierarchy as test_module

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def load_test_hierarchy():
    return test_module.Hierarchy.load_json(os.path.join(DATA_PATH, 'hierarchy.json'))


@raises(KeyError)
def test_find_in_hierarchy_0():
    eq_(test_module.Hierarchy({}).find('id', 'xxx'), [])


def test_find_in_hierarchy_1():
    eq_(test_module.Hierarchy({"id": 997, "children": []}).find('id', 'xxx'), [])


def test_find_in_hierarchy_beyond_me():
    found = test_module.Hierarchy({"id": 997, "prop": "a", "children": []}).find('prop', 'a')
    eq_(len(found), 1)

    found = test_module.Hierarchy(
        {"id": 997, "prop": "a", "children": [
            {"id": 998, "prop": "a", "children": []}
        ]}).find('prop', 'a')

    eq_(len(found), 2)


def test_find_in_hierarchy_2():
    res = load_test_hierarchy().find('acronym', 'L1')
    eq_(len(res), 1)
    eq_(res[0].data['acronym'], 'L1')


@raises(KeyError)
def test_get_in_hierarchy_0():
    test_module.Hierarchy({}).get('id')


def test_get_in_hierarchy_1():
    eq_(test_module.Hierarchy({"id": 997, "children": []}).get('id'), set([997]))


def test_get_in_hierarchy_2():
    h = load_test_hierarchy().find('acronym', 'S1HL')
    res = h[0].get('id')
    eq_(res, set([726, 1125, 1126, 1127, 1128, 1129, 1130]))


def test_collect_in_hierarchy():
    h = test_module.Hierarchy(
        {"id": 997, "prop": "a", "prop2": "a", "children": [
            {"id": 998, "prop": "a", "prop2": "b", "children": []}
        ]})

    eq_(h.collect('prop', 'a', 'prop2'), set(['a', 'b']))


def test_print_hierarchy_empty():
    h = test_module.Hierarchy({})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = test_module.Hierarchy({'children': []})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = test_module.Hierarchy({'children': [{}]})
    eq_(str(h), '<unnamed section>\n'
                '    children: [\n'
                '    <unnamed section>\n'
                '        children: []\n'
                '    ]')


def test_print_hierarchy_props():
    h = test_module.Hierarchy({'name': 'brains', 'prop': 'a'})
    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: []')

    h = test_module.Hierarchy({'name': 'brains', 'prop': 'a',
                        'children': [{'name': 'grey stuff', 'prop': 'b'}]})

    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: [\n'
                '    grey stuff\n'
                '        prop: b\n'
                '        children: []\n'
                '    ]')
