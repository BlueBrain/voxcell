import numpy as np
from numpy.testing import assert_equal
import tempfile
from nose.tools import eq_, raises
import os

from brainbuilder.utils import genbrain as gb

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_read_mhd():
    got = gb.read_mhd(os.path.join(DATA_PATH, 'atlasVolume.mhd'))

    expected = {
        'AnatomicalOrientation': '???',
        'BinaryData': True,
        'BinaryDataByteOrderMSB': False,
        'CenterOfRotation': np.array([0, 0, 0]),
        'DimSize': np.array([52, 32, 45]),
        'ElementDataFile': 'atlasVolume.raw',
        'ElementSpacing': np.array([25, 25, 25]),
        'ElementType': 'MET_UCHAR',
        'NDims': 3,
        'ObjectType': 'Image',
        'Offset': np.array([0, 0, 0]),
        'TransformMatrix': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    }

    eq_(set(got.keys()), set(expected.keys()))

    for k in got.keys():
        v0, v1 = got[k], expected[k]
        if isinstance(v0, np.ndarray):
            assert_equal(v0, v1)
        else:
            eq_(v0, v1)


def test_save_mhd():

    original_path = os.path.join(DATA_PATH, 'atlasVolume.mhd')
    mhd = gb.read_mhd(original_path)

    with tempfile.NamedTemporaryFile() as f:
        gb.save_mhd(f.name, mhd)
        f.seek(0)
        contents = f.read()

        with open(original_path) as o:
            original_contents = o.read()

        eq_(contents, original_contents)


def test_save_metaio():

    original_path = os.path.join(DATA_PATH, 'atlasVolume.mhd')
    vd = gb.VoxelData.load_metaio(original_path)

    with tempfile.NamedTemporaryFile(suffix='.mhd') as f:
        vd.save_metaio(f.name)
        f.seek(0)

        # compare MHD
        contents = dict(l.split(' = ') for l in f.read().split('\n') if l)

        with open(original_path) as o:
            original_contents = dict(l.split(' = ') for l in o.read().split('\n') if l)

        eq_(sorted(contents.keys()), sorted(original_contents.keys()))

        # compare RAW
        rawname = contents['ElementDataFile']
        original_rawname = os.path.join(DATA_PATH, original_contents['ElementDataFile'])
        with open(rawname) as r:
            with open(original_rawname) as ro:
                eq_(r.read(), ro.read())


def test_load_raw_float():
    original = np.random.random((3, 4, 5)).astype(np.float32)

    with tempfile.NamedTemporaryFile() as f:
        original.transpose().tofile(f.name)

        restored = gb.load_raw('MET_FLOAT', (3, 4, 5), f.name)
        assert np.all(original == restored)


def test_load_raw_int():
    original = (np.random.random((3, 4, 5)) * 10).astype(np.uint8)

    with tempfile.NamedTemporaryFile() as f:
        original.transpose().tofile(f.name)

        restored = gb.load_raw('MET_UCHAR', (3, 4, 5), f.name)
        assert np.all(original == restored)


def test_load_meta_io():
    original = gb.VoxelData.load_metaio(os.path.join(DATA_PATH, 'atlasVolume.mhd'),
                                        os.path.join(DATA_PATH, 'atlasVolume.raw'))

    with tempfile.NamedTemporaryFile() as f:
        original.raw.transpose().tofile(f.name)

        restored = gb.VoxelData.load_metaio(os.path.join(DATA_PATH, 'atlasVolume.mhd'), f.name)
        assert np.all(original.raw == restored.raw)


def test_cell_density_from_positions_0():
    positions = np.zeros((0, 3))
    density_dimensions = (3, 4, 5)
    voxel_dimensions = (25, 25, 25)

    result = gb.cell_density_from_positions(positions, density_dimensions, voxel_dimensions)

    assert_equal(result.raw, np.zeros(density_dimensions))


def test_cell_density_from_positions_1():
    positions = np.zeros((1, 3))
    density_dimensions = (3, 4, 5)
    voxel_dimensions = (25, 25, 25)

    result = gb.cell_density_from_positions(positions, density_dimensions, voxel_dimensions)

    expected = np.zeros(density_dimensions)
    expected[0, 0, 0] = 1
    assert_equal(result.raw, expected)


def test_cell_density_from_positions_homogeneous():
    density_dimensions = (3, 4, 5)

    positions = np.zeros((np.prod(density_dimensions), 3))
    idx = 0
    for i in xrange(density_dimensions[0]):
        for j in xrange(density_dimensions[1]):
            for k in xrange(density_dimensions[2]):
                positions[idx] = [i, j, k]
                idx += 1

    voxel_dimensions = (1, 1, 1)

    result = gb.cell_density_from_positions(positions, density_dimensions, voxel_dimensions)

    assert_equal(result.raw, np.ones(density_dimensions))


def test_load_hierarchy_0():
    with tempfile.NamedTemporaryFile() as f:
        f.write('{"msg": [{}]}')
        f.flush()

        h = gb.Hierarchy.load(f.name)
        eq_(h.data, {})
        eq_(h.children, [])


def test_load_hierarchy_1():
    h = gb.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json'))
    eq_(h.data['name'], 'root')


@raises(KeyError)
def test_find_in_hierarchy_0():
    eq_(gb.Hierarchy({}).find('id', 'xxx'), [])


def test_find_in_hierarchy_1():
    eq_(gb.Hierarchy({"id": 997, "children": []}).find('id', 'xxx'), [])


def test_find_in_hierarchy_beyond_me():
    found = gb.Hierarchy({"id": 997, "prop": "a", "children": []}).find('prop', 'a')
    eq_(len(found), 1)

    found = gb.Hierarchy(
        {"id": 997, "prop": "a", "children": [
            {"id": 998, "prop": "a", "children": []}
        ]}).find('prop', 'a')

    eq_(len(found), 2)


def test_find_in_hierarchy_2():
    res = gb.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json')).find(
                               'name', 'Primary somatosensory area, barrel field')
    eq_(len(res), 1)
    eq_(res[0].data['name'], 'Primary somatosensory area, barrel field')


@raises(KeyError)
def test_get_in_hierarchy_0():
    gb.Hierarchy({}).get('id')


def test_get_in_hierarchy_1():
    eq_(gb.Hierarchy({"id": 997, "children": []}).get('id'), set([997]))


def test_get_in_hierarchy_2():
    h = gb.Hierarchy.load(os.path.join(DATA_PATH, 'hierarchy.json')).find(
                             'name', 'Primary somatosensory area, barrel field')
    res = h[0].get('id')
    eq_(res, set([329, 981, 201, 1047, 1070, 1038, 1062]))


def test_collect_in_hierarchy():
    h = gb.Hierarchy(
        {"id": 997, "prop": "a", "prop2": "a", "children": [
            {"id": 998, "prop": "a", "prop2": "b", "children": []}
        ]})

    eq_(h.collect('prop', 'a', 'prop2'), set(['a', 'b']))


def test_print_hierarchy_empty():
    h = gb.Hierarchy({})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = gb.Hierarchy({'children': []})
    eq_(str(h), '<unnamed section>\n'
                '    children: []')

    h = gb.Hierarchy({'children': [{}]})
    eq_(str(h), '<unnamed section>\n'
                '    children: [\n'
                '    <unnamed section>\n'
                '        children: []\n'
                '    ]')


def test_print_hierarchy_props():
    h = gb.Hierarchy({'name': 'brains', 'prop': 'a'})
    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: []')

    h = gb.Hierarchy({'name': 'brains', 'prop': 'a',
                      'children': [{'name': 'grey stuff', 'prop': 'b'}]})

    eq_(str(h), 'brains\n'
                '    prop: a\n'
                '    children: [\n'
                '    grey stuff\n'
                '        prop: b\n'
                '        children: []\n'
                '    ]')


def test_create_voxel_cube():
    shape = (10, 10, 10)
    ret = gb.create_voxel_cube(*shape)
    assert_equal(ret.shape, shape)
    eq_(len(np.nonzero(ret)[0]), 0)
    eq_(len(np.nonzero(np.logical_not(ret))[0]), 10*10*10)


def test_gen_indices():
    z = np.zeros(shape=(10, 10, 10))
    ret = list(gb.gen_indices(z))
    eq_(len(ret), 10*10*10)
    eq_(ret[0], (0, 0, 0))


def test_get_popular_voxels():
    shape = (10, 10, 10)
    ret = gb.create_voxel_cube(*shape)
    ret[0, 0, 0] = [1, 2, 3, 4, 5]
    ret[0, 0, 1] = [1, 2, 3, 4]
    popular = list(gb.get_popular_voxels(ret))
    eq_(popular[0], (5, (0, 0, 0)))
    eq_(popular[1], (4, (0, 0, 1)))


def test_build_sphere_mask_odd_0():
    assert_equal(gb.build_sphere_mask((3, 3), 0),
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_1():
    assert_equal(gb.build_sphere_mask((3, 3), 1),
                 np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_2():
    assert_equal(gb.build_sphere_mask((3, 3), 1.25),
                 np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_3():
    assert_equal(gb.build_sphere_mask((11, 9), 1),
                 np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_odd_4():
    mask = gb.build_sphere_mask((1001, 901), 1)
    expected = np.zeros((1001, 901), dtype=np.bool)
    expected[500, 450] = True
    assert_equal(mask, expected)


def test_build_sphere_mask_even_0():
    assert_equal(gb.build_sphere_mask((4, 4), 0),
                 np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_1():
    assert_equal(gb.build_sphere_mask((4, 4), 1),
                 np.array([[0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_2():
    assert_equal(gb.build_sphere_mask((4, 4), 1.25),
                 np.array([[0, 1, 0, 0],
                           [1, 1, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0]], dtype=np.bool))


def test_build_sphere_mask_even_3():
    mask = gb.build_sphere_mask((1002, 902), 1)
    expected = np.zeros((1002, 902), dtype=np.bool)
    expected[500, 450] = True
    assert_equal(mask, expected)


def test_build_homogeneous_density_empty():
    assert_equal(gb.build_homogeneous_density(np.array([], dtype=np.bool), ()).raw,
                 np.array([]))


def test_build_homogeneous_density_1():
    assert_equal(gb.build_homogeneous_density(np.array([0, 1, 0], dtype=np.bool), (1,)).raw,
                 np.array([0, 255, 0]))


def test_get_regions_mask_by_ids_0():
    assert_equal(gb.get_regions_mask_by_ids(np.ones((3, 3)), set([1])),
                 np.ones((3, 3), dtype=np.bool))

    assert_equal(gb.get_regions_mask_by_ids(np.ones((3, 3)), set([3])),
                 np.zeros((3, 3), dtype=np.bool))

