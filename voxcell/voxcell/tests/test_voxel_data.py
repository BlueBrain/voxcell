import os
import tempfile
import nose.tools as nt
from nose.tools import eq_, ok_

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

import voxcell.voxel_data as test_module
from voxcell.exceptions import VoxcellError


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_clip():
    raw = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6), offset=(10, 20))
    clipped = original.clip(bbox=((11, 25), (15, 33)))

    assert_equal(clipped.raw, [[1]])
    assert_equal(clipped.voxel_dimensions, original.voxel_dimensions)
    assert_equal(clipped.offset, (12, 26))

    # check that they are independent
    raw[1, 1] = 2
    assert_equal(clipped.raw, [[1]])


def test_clip_inplace():
    raw = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6), offset=(10, 20))
    original.clip(bbox=((11, 25), (15, 33)), inplace=True)

    assert_equal(original.raw, [[1]])
    assert_equal(original.offset, (12, 26))


def test_clip_empty():
    raw = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6), offset=(10, 20))
    assert_raises(
        VoxcellError,
        original.clip,
        bbox=((10, 10), (10, 10))
    )


def test_clip_out_of_bounds():
    raw = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6), offset=(10, 20))
    clipped = original.clip(bbox=((-1000, -1000), (1000, 1000)))
    assert_equal(clipped.raw, original.raw)
    assert_equal(clipped.voxel_dimensions, original.voxel_dimensions)
    assert_equal(clipped.offset, original.offset)


def test_lookup():
    raw = np.array([[11, 12], [21, 22]])
    v = test_module.VoxelData(raw, (2, 3), offset=np.array([2, 2]))
    assert_equal(v.lookup([[2, 3]]), [11])
    assert_equal(v.lookup([[2, 10], [1, 5]], outer_value=42), [42, 42])
    assert_equal(v.lookup([[2, 10], [1, 5], [4, 5]], outer_value=42), [42, 42, 22])
    assert_raises(VoxcellError, v.lookup, [[1, 5]])
    assert_raises(VoxcellError, v.lookup, [[2, 10]])


def test_lookup_vector_data():
    raw = np.array([[[11], [12]], [[21], [22]]])
    v = test_module.VoxelData(raw, (2, 2))
    assert_equal(v.lookup([[1, 1], [3, 3]]), [[11], [22]])


def test_positions_to_indices():
    raw = np.zeros(2)
    v = test_module.VoxelData(raw, voxel_dimensions=(10.0,), offset=(10.0,))
    assert_equal(v.positions_to_indices([15., 25.]), [0, 1])
    # border effects
    assert_equal(v.positions_to_indices([9.9999999]), [0])
    assert_equal(v.positions_to_indices([19.9999999]), [0])

def test_load_nrrd_scalar_payload():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'scalar.nrrd'))
    eq_(actual.raw.shape, (1, 2))
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])

def test_load_nrrd_vector_payload():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'vector.nrrd'))
    eq_(actual.raw.shape, (1, 2, 3))
    assert_almost_equal(actual.raw, [[[11, 12, 13], [21, 22, 23]]])
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])

def test_load_nrrd_with_space_directions():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'space_directions.nrrd'))
    eq_(actual.raw.shape, (1, 2, 3))
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])

def test_load_nrrd_fail():
    # no spacing information
    assert_raises(VoxcellError, test_module.VoxelData.load_nrrd, os.path.join(DATA_PATH, 'no_spacings_fail.nrrd'))
    # space directions is non-diagonal
    assert_raises(NotImplementedError, test_module.VoxelData.load_nrrd, os.path.join(DATA_PATH, 'space_directions_fail.nrrd'))

def test_save_nrrd():
    ''' test saving a test nrrd file and check basic attributes '''
    vd = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'vector.nrrd'))
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        new = test_module.VoxelData.load_nrrd(f.name)
        ok_(np.allclose(vd.raw, new.raw))
        ok_(np.allclose(vd.voxel_dimensions, new.voxel_dimensions))
        ok_(np.allclose(vd.offset, new.offset))


def test_shape_checks():
    raw = np.zeros(shape=(2, 2))
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=[[25], [25]])
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=(25, 25), offset=(0,))
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=(25, 25, 25))


def test_with_data():
    original = test_module.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    replaced = original.with_data(np.array([2, 3]))
    assert_equal(replaced.raw, [2, 3])
    assert_equal(replaced.voxel_dimensions, original.voxel_dimensions)
    assert_equal(replaced.offset, original.offset)


def test_indices_to_positions():
    vd = test_module.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    positions = vd.indices_to_positions(np.array([0, 0.5, 1]))
    assert_almost_equal(positions, [42, 43, 44])


def test_count():
    vd = test_module.VoxelData(np.array([0, 1, 1, 2]), voxel_dimensions=(2,))
    assert_equal(vd.count(7), 0)
    assert_equal(vd.count(1), 2)
    assert_equal(vd.count([0, 2]), 2)
    assert_equal(vd.count(set([0, 2])), 2)


def test_volume():
    vd = test_module.VoxelData(np.array([[0, 1], [1, 2]]), voxel_dimensions=(2, -3))
    assert_equal(vd.volume(1), 12)
    assert_equal(vd.volume(13), 0)


def test_filter():
    raw = np.array([[11, 12], [21, 22]])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6), offset=(10, 20))
    filtered = original.filter(lambda p: p[0] > 12 and p[1] > 26)
    assert_equal(original.raw, raw)
    assert_equal(filtered.raw, [[0, 0], [0, 22]])
    assert_equal(filtered.raw, [[0, 0], [0, 22]])
    assert_equal(filtered.offset, original.offset)
    assert_equal(filtered.voxel_dimensions, original.voxel_dimensions)


def test_filter_inplace():
    original = test_module.VoxelData(np.array([[11, 12], [21, 22]]), (2, 6), offset=(10, 20))
    original.filter(lambda p: p[0] > 12 and p[1] > 26, inplace=True)
    assert_equal(original.raw, [[0, 0], [0, 22]])


def test_orientation_field():
    field = test_module.OrientationField(np.array([[0., 0., 0., 1.]]), voxel_dimensions=(2,))
    npt.assert_almost_equal(
        field.lookup([1.]),
        [np.identity(3)]
    )

def test_orientation_field_compact():
    field = test_module.OrientationField(np.array([[0, 0, 0, 127]], dtype=np.int8), voxel_dimensions=(2,))
    npt.assert_almost_equal(
        field.lookup([1.]),
        [np.identity(3)]
    )

def test_orientation_field_raises():
    nt.assert_raises(
        VoxcellError,
        test_module.OrientationField, np.zeros(4), voxel_dimensions=(1,)
    )
    nt.assert_raises(
        VoxcellError,
        test_module.OrientationField, np.zeros((3, 3)), voxel_dimensions=(1,)
    )
