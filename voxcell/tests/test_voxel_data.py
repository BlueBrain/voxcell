import os
import tempfile
from nose.tools import eq_, ok_

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from voxcell import core, VoxcellError

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_read_mhd():
    got = core.read_mhd(os.path.join(DATA_PATH, 'atlasVolume.mhd'))

    expected = {
        'AnatomicalOrientation': '???',
        'BinaryData': True,
        'BinaryDataByteOrderMSB': False,
        'CenterOfRotation': np.array([0.0, 0.0, 0]),
        'DimSize': np.array([52, 32, 45]),
        'ElementDataFile': 'atlasVolume.raw',
        'ElementSpacing': np.array([25.0, 25, 25]),
        'ElementType': 'MET_UCHAR',
        'NDims': 3,
        'ObjectType': 'Image',
        'Offset': np.array([0.0, 0, 0]),
        'TransformMatrix': np.array([1.0, 0, 0, 0, 1, 0, 0, 0, 1])
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
    mhd = core.read_mhd(original_path)

    with tempfile.NamedTemporaryFile() as f:
        core.save_mhd(f.name, mhd)
        f.seek(0)
        contents = f.read()

        with open(original_path) as o:
            original_contents = o.read()

        eq_(contents, original_contents)


def test_save_metaio():

    original_path = os.path.join(DATA_PATH, 'atlasVolume.mhd')
    vd = core.VoxelData.load_metaio(original_path)

    with tempfile.NamedTemporaryFile(suffix='.mhd') as f:
        vd.save_metaio(f.name)
        f.seek(0)

        # compare MHD
        contents = dict(l.split(' = ') for l in f.read().split('\n') if l)

        with open(original_path) as o:
            original_contents = dict(l.split(' = ') for l in o.read().split('\n') if l)

        eq_(sorted(contents.keys()), sorted(original_contents.keys()))

        # compare RAW
        rawname = os.path.join(os.path.dirname(f.name), contents['ElementDataFile'])
        original_rawname = os.path.join(DATA_PATH, original_contents['ElementDataFile'])
        with open(rawname) as r:
            with open(original_rawname) as ro:
                eq_(r.read(), ro.read())


def test_load_raw_float():
    original = np.random.random((3, 4, 5)).astype(np.float32)

    with tempfile.NamedTemporaryFile() as f:
        original.transpose().tofile(f.name)

        restored = core.load_raw('MET_FLOAT', (3, 4, 5), f.name)
        assert np.all(original == restored)


def test_load_raw_int():
    original = (np.random.random((3, 4, 5)) * 10).astype(np.uint8)

    with tempfile.NamedTemporaryFile() as f:
        original.transpose().tofile(f.name)

        restored = core.load_raw('MET_UCHAR', (3, 4, 5), f.name)
        assert np.all(original == restored)


def test_load_meta_io():
    original = core.VoxelData.load_metaio(os.path.join(DATA_PATH, 'atlasVolume.mhd'),
                                          os.path.join(DATA_PATH, 'atlasVolume.raw'))

    with tempfile.NamedTemporaryFile() as f:
        original.raw.transpose().tofile(f.name)

        restored = core.VoxelData.load_metaio(os.path.join(DATA_PATH, 'atlasVolume.mhd'), f.name)
        assert np.all(original.raw == restored.raw)


def test_clip_volume():
    raw = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

    aabb = (np.array([1, 1]), np.array([1, 1]))

    v = core.VoxelData(raw, (1, 1))
    r = v.clipped(aabb)

    assert_equal(r.raw, np.array([[1]]))

    # check that they are independent
    raw[1, 1] = 2
    eq_(r.raw[0, 0], 1)


def test_load_nrrd():
    ''' test loading a test nrrd file and check basic attributes '''
    got = core.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'test.nrrd'))
    eq_(got.raw.shape, (528, 320, 456))
    assert_almost_equal(got.voxel_dimensions, [42, 43, 44])
    assert_equal(got.offset, np.array([0, 0, 0]))

def test_load_nrrd_fail():
    ''' test loading a test nrrd file without 'spacings' attribute '''
    assert_raises(VoxcellError, core.VoxelData.load_nrrd, os.path.join(DATA_PATH, 'test_fail.nrrd'))

def test_save_nrrd():
    ''' test saving a test nrrd file and check basic attributes '''
    vd = core.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'test.nrrd'))
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        new = core.VoxelData.load_nrrd(f.name)
        ok_(np.allclose(vd.raw, new.raw))
        ok_(np.allclose(vd.voxel_dimensions, new.voxel_dimensions))
        ok_(np.allclose(vd.offset, new.offset))

def test_roundtrip_nrrd():
    vd = core.VoxelData.load_metaio(os.path.join(DATA_PATH, 'atlasVolume.mhd'),
                                    os.path.join(DATA_PATH, 'atlasVolume.raw'))

    with tempfile.NamedTemporaryFile() as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        new = core.VoxelData.load_nrrd(f.name)
        ok_(np.allclose(vd.raw, new.raw))
        ok_(np.allclose(vd.voxel_dimensions, new.voxel_dimensions))
        ok_(np.allclose(vd.offset, new.offset))
