import numpy as np
from numpy.testing import assert_equal
import tempfile
from nose.tools import eq_, raises
import os

from brainbuilder.utils import genbrain as gb

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_read_mhd():
    eq_(gb.read_mhd(os.path.join(DATA_PATH, 'atlasVolume.mhd')),
        {'AnatomicalOrientation': '???',
         'BinaryData': True,
         'BinaryDataByteOrderMSB': False,
         'CenterOfRotation': (0, 0, 0),
         'DimSize': (52, 32, 45),
         'ElementDataFile': 'atlasVolume.raw',
         'ElementSpacing': (25, 25, 25),
         'ElementType': 'MET_UCHAR',
         'NDims': 3,
         'ObjectType': 'Image',
         'Offset': (0, 0, 0),
         'TransformMatrix': (1, 0, 0, 0, 1, 0, 0, 0, 1)})


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
    original = gb.MetaIO.load(os.path.join(DATA_PATH, 'atlasVolume.mhd'),
                              os.path.join(DATA_PATH, 'atlasVolume.raw'))

    with tempfile.NamedTemporaryFile() as f:
        original.raw.transpose().tofile(f.name)

        restored = gb.MetaIO.load(os.path.join(DATA_PATH, 'atlasVolume.mhd'), f.name)
        assert np.all(original.raw == restored.raw)


def test_cell_density_from_positions_0():
    positions = np.zeros((0, 3))
    density_dimensions = (3, 4, 5)
    voxel_dimensions = (25, 25, 25)

    result = gb.cell_density_from_positions(positions, density_dimensions, voxel_dimensions)

    assert_equal(result, np.zeros(density_dimensions))


def test_cell_density_from_positions_1():
    positions = np.zeros((1, 3))
    density_dimensions = (3, 4, 5)
    voxel_dimensions = (25, 25, 25)

    result = gb.cell_density_from_positions(positions, density_dimensions, voxel_dimensions)

    expected = np.zeros(density_dimensions)
    expected[0, 0, 0] = 1
    assert_equal(result, expected)


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

    assert_equal(result, np.ones(density_dimensions))


def test_load_hierarchy_0():
    with tempfile.NamedTemporaryFile() as f:
        f.write('{}')
        f.flush()

        h = gb.load_hierarchy(f.name)
        eq_(h, {})


def test_load_hierarchy_1():
    h = gb.load_hierarchy(os.path.join(DATA_PATH, 'hierarchy.json'))
    eq_(h['name'], 'root')


@raises(KeyError)
def test_find_in_hierarchy_0():
    eq_(gb.find_in_hierarchy({}, 'id', 'xxx'), [])


def test_find_in_hierarchy_1():
    eq_(gb.find_in_hierarchy({ "id": 997, "children": []}, 'id', 'xxx'), [])


def test_find_in_hierarchy_2():
    res = gb.find_in_hierarchy(gb.load_hierarchy(os.path.join(DATA_PATH, 'hierarchy.json')),
                               'name', 'Primary somatosensory area, barrel field')
    eq_(len(res), 1)
    eq_(res[0]['name'], 'Primary somatosensory area, barrel field')


@raises(KeyError)
def test_get_in_hierarchy_0():
    gb.get_in_hierarchy({}, 'id')


def test_get_in_hierarchy_1():
    eq_(gb.get_in_hierarchy({"id": 997, "children": []}, 'id'), [997])


def test_get_in_hierarchy_2():
    h = gb.find_in_hierarchy(gb.load_hierarchy(os.path.join(DATA_PATH, 'hierarchy.json')),
                             'name', 'Primary somatosensory area, barrel field')
    res = gb.get_in_hierarchy(h[0], 'id')
    eq_(res, [329, 981, 201, 1047, 1070, 1038, 1062])


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
    assert_equal(gb.build_homogeneous_density(np.array([0, 1, 0], dtype=np.bool), (1,), 42).raw,
                 np.array([0, 42, 0]))
