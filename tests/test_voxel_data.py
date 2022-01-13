import operator
import os
import re
import tempfile
from unittest.mock import Mock, call

import nrrd
import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_raises

import voxcell.voxel_data as test_module
from voxcell.exceptions import VoxcellError

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def test_lookup():
    raw = np.array([[11, 12], [21, 22]])
    v = test_module.VoxelData(raw, (2, 3), offset=np.array([2, 2]))
    assert v.lookup([[2, 3]]) == [11]
    assert_array_equal(v.lookup([[2, 10], [1, 5]], outer_value=42), [42, 42])
    assert_array_equal(v.lookup([[2, 10], [1, 5], [4, 5]], outer_value=42), [42, 42, 22])
    assert_raises(VoxcellError, v.lookup, [[1, 5]])
    assert_raises(VoxcellError, v.lookup, [[2, 10]])


def test_lookup_vector_data():
    raw = np.array([[[11], [12]], [[21], [22]]])
    v = test_module.VoxelData(raw, (2, 2))
    assert_array_equal(v.lookup([[1, 1], [3, 3]]), [[11], [22]])


def test_positions_to_indices():
    raw = np.zeros(2)
    v = test_module.VoxelData(raw, voxel_dimensions=(10.0,), offset=(10.0,))
    assert_array_equal(v.positions_to_indices([15., 25.]), [0, 1])
    assert_array_equal(v.positions_to_indices([15., 25.], keep_fraction=True), [0.5, 1.5])

    # border effects
    assert v.positions_to_indices([9.9999999]) == [0]
    assert v.positions_to_indices([19.9999999]) == [0]


def test_load_nrrd_scalar_payload():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'scalar.nrrd'))
    assert actual.raw.shape == (1, 2)
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])
    assert_almost_equal(actual.bbox, np.array([[100, 200], [110, 240]]))


def test_load_nrrd_vector_payload():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'vector.nrrd'))
    assert actual.raw.shape == (1, 2, 3)
    assert_almost_equal(actual.raw, [[[11, 12, 13], [21, 22, 23]]])
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])


def test_load_nrrd_with_space_directions():
    actual = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'space_directions.nrrd'))
    assert actual.raw.shape == (1, 2, 3)
    assert_almost_equal(actual.voxel_dimensions, [10, 20])
    assert_almost_equal(actual.offset, [100, 200])


def test_load_nrrd_fail():
    # no spacing information
    assert_raises(
        VoxcellError,
        test_module.VoxelData.load_nrrd,
        os.path.join(DATA_PATH, 'no_spacings_fail.nrrd'),
    )
    # space directions is non-diagonal
    assert_raises(
        NotImplementedError,
        test_module.VoxelData.load_nrrd,
        os.path.join(DATA_PATH, 'space_directions_fail.nrrd'),
    )


def test_save_nrrd():
    """Test saving a test nrrd file and check basic attributes."""
    vd = test_module.VoxelData.load_nrrd(os.path.join(DATA_PATH, 'vector.nrrd'))
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        new = test_module.VoxelData.load_nrrd(f.name)
        assert np.allclose(vd.raw, new.raw)
        assert np.allclose(vd.voxel_dimensions, new.voxel_dimensions)
        assert np.allclose(vd.offset, new.offset)
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name, encoding='raw')
        f.flush()
        f.seek(0)
        new = test_module.VoxelData.load_nrrd(f.name)
        assert np.allclose(vd.raw, new.raw)


def test_save_nrrd_with_extra_axes():
    """Test saving a numpy array with more than 3 dimensions."""
    raw = np.zeros((6,7,8,4,3)) # two extra dimensions
    vd = test_module.VoxelData(raw, (1.0, 2.0, 3.0))
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        _, header = nrrd.read(f.name)
        # pynrrd will convert None into np.array([np.nan, np.nan, np.nan])
        assert_array_equal(header['space directions'],
            [
                [np.nan] * 3, [np.nan] * 3, (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)
            ])
        assert 'kinds' not in header


def test_save_nrrd_vector_field():
    """Test saving a numpy array with exactly 4 dimensions and a numeric dtype."""
    raw = np.zeros((6, 7, 8, 5)) # one extra dimension
    vd = test_module.VoxelData(raw, (1.0, 2.0, 3.0))
    with tempfile.NamedTemporaryFile(suffix='.nrrd') as f:
        vd.save_nrrd(f.name)
        f.flush()
        f.seek(0)
        _, header = nrrd.read(f.name)
        # pynrrd's reader will convert None into np.array([np.nan, np.nan, np.nan])
        assert_array_equal (header['space directions'],
            [
                [np.nan] * 3, (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)
            ])
        assert_array_equal(header['kinds'], ['vector', 'domain', 'domain', 'domain'])


def test_shape_checks():
    raw = np.zeros(shape=(2, 2))
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=[[25], [25]])
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=(25, 25), offset=(0,))
    assert_raises(VoxcellError, test_module.VoxelData, raw, voxel_dimensions=(25, 25, 25))


def test_with_data():
    original = test_module.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    replaced = original.with_data(np.array([2, 3]))
    assert_array_equal(replaced.raw, [2, 3])
    assert replaced.voxel_dimensions == original.voxel_dimensions
    assert replaced.offset == original.offset


def test_indices_to_positions():
    vd = test_module.VoxelData(np.array([0, 1]), voxel_dimensions=(2,), offset=(42,))
    positions = vd.indices_to_positions(np.array([0, 0.5, 1]))
    assert_almost_equal(positions, [42, 43, 44])


def test_count():
    vd = test_module.VoxelData(np.array([0, 1, 1, 2]), voxel_dimensions=(2,))
    assert vd.count(7) == 0
    assert vd.count(1) == 2
    assert vd.count([0, 2]) == 2
    assert vd.count(set([0, 2])) == 2


def test_volume():
    vd = test_module.VoxelData(np.array([[0, 1], [1, 2]]), voxel_dimensions=(2, -3))
    assert vd.volume(1) == 12
    assert vd.volume(13) == 0


def test_clip():
    raw = np.array([1, 2, 3])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    clipped = original.clip(bbox=((11,), (15,)), na_value=-1)
    assert_array_equal(original.raw, raw)
    assert_array_equal(clipped.raw, [-1, 2, -1])
    assert clipped.voxel_dimensions == original.voxel_dimensions
    assert clipped.offset == original.offset


def test_clip_inplace():
    raw = np.array([1, 2, 3])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    original.clip(bbox=((11,), (15,)), na_value=-1, inplace=True)
    assert_array_equal(original.raw, [-1, 2, -1])


def test_clip_empty():
    raw = np.array([1, 2, 3])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    assert_raises(
        VoxcellError,
        original.clip,
        bbox=((10,), (10,))
    )


def test_clip_out_of_bounds():
    raw = np.array([1, 2, 3])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    clipped = original.clip(bbox=((-1000,), (1000,)))
    assert_array_equal(clipped.raw, original.raw)


def test_filter():
    raw = np.array([[[11], [12]], [[21], [22]]])
    original = test_module.VoxelData(raw, voxel_dimensions=(2, 6, 10), offset=(10, 20, 30))
    filtered = original.filter(lambda p: np.logical_and(p[:, 0] > 12, p[:, 1] > 26))
    assert_array_equal(original.raw, raw)
    assert_array_equal(filtered.raw, [[[0], [0]], [[0], [22]]])
    assert_array_equal(filtered.offset, original.offset)
    assert_array_equal(filtered.voxel_dimensions, original.voxel_dimensions)


def test_filter_inplace():
    original = test_module.VoxelData(np.array([[[11], [12]], [[21], [22]]]), (2, 6, 10), offset=(10, 20, 30))
    original.filter(lambda p: np.logical_and(p[:, 0] > 12, p[:, 1] > 26), inplace=True)
    assert_array_equal(original.raw, [[[0], [0]], [[0], [22]]])


def test_compact():
    raw = np.array([0, 42, -1])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    compact = original.compact(na_values=(0, -1))
    assert_array_equal(original.raw, raw)
    assert original.offset == 10
    assert compact.raw == raw[1:2]
    assert compact.voxel_dimensions == original.voxel_dimensions
    assert compact.offset == 12


def test_compact_inplace():
    raw = np.array([0, 42, -1])
    original = test_module.VoxelData(raw, voxel_dimensions=(2,), offset=(10,))
    original.compact(na_values=(0, -1), inplace=True)
    assert original.raw == raw[1:2]
    assert original.offset == 12


def test_orientation_field():
    field = test_module.OrientationField(np.array([[1., 0., 0., 0.]]), voxel_dimensions=(2,))
    npt.assert_almost_equal(
        field.lookup([1.]),
        [np.identity(3)]
    )


def test_orientation_field_compact():
    field = test_module.OrientationField(np.array([[127, 0, 0, 0]], dtype=np.int8), voxel_dimensions=(2,))
    npt.assert_almost_equal(
        field.lookup([1.]),
        [np.identity(3)]
    )


def test_orientation_field_raises():
    with pytest.raises(VoxcellError):
        test_module.OrientationField(np.zeros(4), voxel_dimensions=(1,))
    with pytest.raises(VoxcellError):
        test_module.OrientationField(np.zeros((3, 3)), voxel_dimensions=(1,))


def test_roi_mask():
    field = test_module.ROIMask(np.array([1, 0, 0, 0], dtype=np.uint8), voxel_dimensions=(2,))
    actual = field.lookup([[1.], [3]])
    assert actual.dtype == 'bool'
    assert_array_equal(actual, [True, False])


def test_roi_mask_raises():
    with pytest.raises(VoxcellError, match=re.escape("Invalid dtype: 'int64' (expected: '(u)int8")):
        test_module.ROIMask(np.zeros(4, dtype=np.int64), voxel_dimensions=(1,))


def test_values_to_region_attribute():
    region_map = Mock()
    region_map.get.side_effect = lambda _id, attr: {0: "CA1", 1: "SO", 2: "SP"}[_id]
    values = np.array([1, 0, 0, 1])
    actual = test_module.values_to_region_attribute(values, region_map)
    assert region_map.get.call_count == 2  # called once for each different looked up id
    assert region_map.get.call_args_list == [call(0, attr='acronym'), call(1, attr='acronym')]
    assert np.issubdtype(actual.dtype, np.str)
    assert_array_equal(actual, ["SO", "CA1", "CA1", "SO"])


def test_values_to_hemisphere():
    values = np.array([2, 1, 1, 2])
    actual = test_module.values_to_hemisphere(values)
    assert np.issubdtype(actual.dtype, np.str)
    assert_array_equal(actual, ["left", "right", "right", "left"])


def test_values_to_hemisphere_raises_invalid_value():
    values = np.array([2, 1, 1, 99])
    with pytest.raises(VoxcellError, match=re.escape("Invalid values, only [0, 1, 2] are allowed")):
        test_module.values_to_hemisphere(values)


def test_reduce():
    # Add 3 int arrays
    a = test_module.VoxelData(np.array([[11, 12], [21, 22]]), (2, 3))
    b = test_module.VoxelData(np.array([[1, 0], [0, 0]]), (2, 3))
    c = test_module.VoxelData(np.array([[1, 0], [0, 0]]), (2, 3))

    # 1 elements: function is not applied
    d = test_module.VoxelData.reduce(lambda x: x * 100, [a])
    assert_almost_equal(d.raw, [[11, 12], [21, 22]])

    # 2 elements
    e = test_module.VoxelData.reduce(operator.add, [a, b])
    assert_almost_equal(e.raw, [[12, 12], [21, 22]])

    # 3 elements
    f = test_module.VoxelData.reduce(operator.add, [a, b, c])
    assert_almost_equal(f.raw, [[13, 12], [21, 22]])


def test_offset_and_voxel_dimensions_type():
    voxel_data = test_module.VoxelData(
        np.ones((2, 2, 2)), offset=(1, 2, 3), voxel_dimensions=(1, 1, 1))
    assert voxel_data.offset.dtype == np.float32
    assert voxel_data.voxel_dimensions.dtype == np.float32

    voxel_data = test_module.VoxelData(np.ones((2, 2, 2)), voxel_dimensions=(3, 4, 5))
    assert voxel_data.offset.dtype == np.float32
