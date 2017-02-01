import os
import tempfile
from nose.tools import eq_, ok_

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from voxcell import core, VoxcellError

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


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


def test_lookup():
    raw = np.array([[11, 12], [21, 22]])
    v = core.VoxelData(raw, (2, 3), offset=np.array([2, 2]))
    assert_equal(v.lookup([[2, 3]]), [11])
    assert_equal(v.lookup([[2, 10], [1, 5]], outer_value=42), [42, 42])
    assert_equal(v.lookup([[2, 10], [1, 5], [4, 5]], outer_value=42), [42, 42, 22])
    assert_raises(VoxcellError, v.lookup, [[1, 5]])
    assert_raises(VoxcellError, v.lookup, [[2, 10]])


def test_lookup_vector_data():
    raw = np.array([[[11], [12]], [[21], [22]]])
    v = core.VoxelData(raw, (2, 2))
    assert_equal(v.lookup([[1, 1], [3, 3]]), [[11], [22]])


def test_load_nrrd():
    got = core.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'test.nrrd'))
    eq_(got.raw.shape, (528, 320, 456))
    assert_almost_equal(got.voxel_dimensions, [42, 43, 44])
    assert_almost_equal(got.offset, [0, 0, 0])

def test_load_nrrd_with_space_directions():
    got = core.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'test_with_space_directions.nrrd'))
    eq_(got.raw.shape, (409, 608, 286))
    assert_almost_equal(got.voxel_dimensions, [40, 40, -40])
    assert_almost_equal(got.offset, [-8.12492943, -7.91999865, -0.1444])

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


def test_shape_checks():
    raw = np.zeros(shape=(2, 2))
    assert_raises(VoxcellError, core.VoxelData, raw, voxel_dimensions=[[25], [25]])
    assert_raises(VoxcellError, core.VoxelData, raw, voxel_dimensions=(25, 25), offset=(0,))
    assert_raises(VoxcellError, core.VoxelData, raw, voxel_dimensions=(25, 25, 25))


def test_with_data():
    original = core.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    replaced = original.with_data(np.array([2, 3]))
    assert_equal(replaced.raw, [2, 3])
    assert_equal(replaced.voxel_dimensions, original.voxel_dimensions)
    assert_equal(replaced.offset, original.offset)


def test_indices_to_positions():
    vd = core.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    positions = vd.indices_to_positions(np.array([0, 0.5, 1]))
    assert_almost_equal(positions, [42, 43, 44])
