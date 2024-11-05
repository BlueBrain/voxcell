import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal, assert_series_equal

import voxcell.cell_collection as test_module
from voxcell import VoxcellError

DATA_PATH = Path(__file__).parent / 'data'
SONATA_DATA_PATH = DATA_PATH / 'sonata'


def is_cat(series):
    return isinstance(series.dtype, pd.CategoricalDtype)


def euler_to_matrix(bank, attitude, heading):
    """Build 3x3 rotation matrices from arrays of euler angles.

    Based on algorigthm described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm

    Args:
        bank: rotation around X
        attitude: rotation around Z
        heading: rotation around Y
    """

    sa, ca = np.sin(attitude), np.cos(attitude)
    sb, cb = np.sin(bank), np.cos(bank)
    sh, ch = np.sin(heading), np.cos(heading)

    m = np.vstack([ch * ca, -ch * sa * cb + sh * sb, ch * sa * sb + sh * cb,
                   sa, ca * cb, -ca * sb,
                   -sh * ca, sh * sa * cb + ch * sb, -sh * sa * sb + ch * cb]).transpose()

    return m.reshape(m.shape[:-1] + (3, 3))


def test_euler_to_matrix():  # testing the test
    n = 2

    assert_array_equal(
        euler_to_matrix([0] * n, [0] * n, [0] * n),
        np.array([np.diag([1, 1, 1])] * n))

    assert_almost_equal(
        euler_to_matrix([np.deg2rad(90)] * n, [0] * n, [0] * n),
        np.array([np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])] * n))

    assert_almost_equal(
        euler_to_matrix([0] * n, [np.deg2rad(90)] * n, [0] * n),
        np.array([np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])] * n))

    assert_almost_equal(
        euler_to_matrix([0] * n, [0] * n, [np.deg2rad(90)] * n),
        np.array([np.array([[0, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 0]])] * n))


def random_orientations(n):
    return euler_to_matrix(np.random.random(n) * np.pi * 2,
                           np.random.random(n) * np.pi * 2,
                           np.random.random(n) * np.pi * 2)


def random_positions(n):
    return np.random.random((n, 3))


@contextmanager
def tempcwd():
    """Tempdir contextmanager."""
    cwd = os.getcwd()
    dirname = tempfile.mkdtemp(prefix='bbtests_')
    os.chdir(dirname)
    try:
        yield dirname
    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def assert_equal_cells(c0, c1):
    if c0.positions is None:
        assert c1.positions is None
    else:
        assert_almost_equal(c0.positions, c1.positions)
    if c0.orientations is None:
        assert c1.orientations is None
    else:
        assert_almost_equal(c0.orientations, c1.orientations)
    sorted_c0 = c0.properties.sort_index(axis=1)
    sorted_c1 = c1.properties.sort_index(axis=1)
    assert_array_equal(sorted_c0.columns, sorted_c1.columns)
    for column in sorted_c0:
        s_c0, s_c1 = sorted_c0[column], sorted_c1[column]
        if is_cat(s_c0) or is_cat(s_c1):
            assert_array_equal(s_c0.to_numpy(), s_c1.to_numpy())
            assert_array_equal(s_c0.index.to_numpy(), s_c1.index.to_numpy())
        else:
            assert_series_equal(s_c0, s_c1, check_names=True)


def check_roundtrip(original):
    with tempcwd():
        original.save('cells.mvd3')
        restored = test_module.CellCollection.load('cells.mvd3')
        assert_equal_cells(original, restored)
        restored_mvd3 = test_module.CellCollection.load_mvd3('cells.mvd3')
        assert_equal_cells(restored, restored_mvd3)
        original.save('cells.h5')
        restored = test_module.CellCollection.load('cells.h5')
        assert_equal_cells(original, restored)
        restored_sonata = test_module.CellCollection.load_sonata('cells.h5')
        assert_equal_cells(restored, restored_sonata)
        return restored


def test_is_string_enum():
    str_series = pd.Series(list('babc')).astype(CategoricalDtype(list('abcd')))
    assert test_module._is_string_enum(str_series)
    int_series = pd.Series([1, 2, 3, 1], dtype="category")
    assert not test_module._is_string_enum(int_series)
    float_series = pd.Series(pd.Categorical.from_codes(codes=[0, 0, 1, 0], categories=[1.5, -.4]))
    assert not test_module._is_string_enum(float_series)


def test_load_mvd2():
    cells_mvd3 = test_module.CellCollection.load_mvd3(DATA_PATH / 'mvd2_mvd3/circuit.mvd3')
    cells_mvd2 = test_module.CellCollection.load_mvd2(DATA_PATH / 'mvd2_mvd3/circuit.mvd2')
    assert_equal_cells(cells_mvd2, cells_mvd3)


def test_roundtrip_mvd():
    cells = test_module.CellCollection.load_mvd2(DATA_PATH / 'mvd2_mvd3/circuit.mvd2')
    check_roundtrip(cells)


def test_roundtrip_empty():
    cells = test_module.CellCollection()
    check_roundtrip(cells)


def test_roundtrip_none():
    cells = test_module.CellCollection()
    cells.properties['y-factor'] = [0.25, np.nan, 0.75]
    with pytest.raises(VoxcellError):
        check_roundtrip(cells)
    cells.properties['y-factor'] = [None, 0.1, 0.75]
    with pytest.raises(VoxcellError):
        check_roundtrip(cells)


def test_roundtrip_properties_numeric_single():
    cells = test_module.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75]
    check_roundtrip(cells)


def test_roundtrip_properties_numeric_multiple():
    cells = test_module.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75, 0]
    cells.properties['z-factor'] = [0, 0.75, 0.5, 0.25]
    check_roundtrip(cells)


def test_roundtrip_properties_text_single():
    cells = test_module.CellCollection()
    cells.properties['y-type'] = ['pretty', 'ugly', 'pretty']
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()


def test_roundtrip_properties_text_multiple():
    cells = test_module.CellCollection()
    cells.properties['y-type'] = ['ugly', 'pretty', 'ugly', 'ugly']
    cells.properties['z-type'] = ['red', 'blue', 'green', 'alpha']
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()
    restored.properties['z-type'].to_frame()


def test_roundtrip_properties_text_multiple_transform_to_categorical():
    cells = test_module.CellCollection()
    # y-type must be selected as a categorical candidate
    cells.properties['y-type'] = ['ugly', 'ugly', 'ugly', 'ugly', 'pretty']
    cells.properties['z-type'] = ['red', 'blue', 'green', 'alpha', 'optimus_prime']
    # y-type is a string at the beginning
    assert cells.properties['y-type'].dtypes == object
    assert cells.properties['z-type'].dtypes == object
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()
    # y-type should be categorical now
    assert is_cat(restored.properties['y-type'])
    assert cells.properties['z-type'].dtypes == object
    restored.properties['z-type'].to_frame()


def test_roundtrip_positions():
    cells = test_module.CellCollection()
    cells.positions = random_positions(10)
    check_roundtrip(cells)


def test_roundtrip_orientations():
    cells = test_module.CellCollection()
    cells.orientations = random_orientations(10)
    check_roundtrip(cells)


def test_roundtrip_complex():
    cells = test_module.CellCollection()
    n = 5

    cells.positions = random_positions(n)
    cells.orientations = random_orientations(n)
    cells.properties['all_none'] = pd.Categorical.from_codes(codes=[0] * n, categories=[''])
    cells.properties['synapse_class'] = pd.Categorical.from_codes(
        codes=[0, 1, 1, 0, 0], categories=['', 'INH'])
    cells.properties['mtype'] = ['L5_NGC', 'L5_BTC', 'L5_BTC', 'L6_LBC', 'L6_LBC']
    cells.properties['etype'] = ['cADpyr', 'dNAC', 'dNAC', 'bSTUT', 'bSTUT']
    cells.properties['morphology'] = [
        'dend-C250500A-P3_axon-C190898A-P2_-_Scale_x1.000_y1.025_z1.000_-_Clone_2',
        'C240300C1_-_Scale_x1.000_y0.975_z1.000_-_Clone_55',
        'C240300C1_-_Scale_x1.000_y0.975_z1.000_-_Clone_55',
        'dend-Fluo15_right_axon-Fluo2_right_-_Clone_37',
        'dend-Fluo15_right_axon-Fluo2_right_-_Clone_37',
    ]
    prefix = test_module.CellCollection.SONATA_DYNAMIC_PROPERTY
    cells.properties[prefix + 'current'] = np.arange(n)
    cells.properties[prefix + 'some_prop'] = np.random.choice(['t1', 't2'], n)
    check_roundtrip(cells)


def test_remove_unassigned_1():
    cells = test_module.CellCollection()
    n = 5
    positions = random_positions(n)
    orientations = random_orientations(n)
    cells.positions = positions
    cells.orientations = orientations
    cells.properties = pd.DataFrame({
        'foo': ['', 'a', None, 'b', 'c'],
        'bar': [0., None, 2., 3., 4.]
    })
    cells.remove_unassigned_cells()
    assert_array_equal(cells.positions, positions[[0, 3, 4]])
    assert_array_equal(cells.orientations, orientations[[0, 3, 4]])
    assert_frame_equal(
        cells.properties,
        pd.DataFrame({
            'foo': ['', 'b', 'c'],
            'bar': [0., 3., 4.]
        })
    )


def test_remove_unassigned_2():
    cells = test_module.CellCollection()
    n = 2
    cells.positions = random_positions(n)
    cells.orientations = random_orientations(n)
    cells.properties = pd.DataFrame({
        'foo': ['a', 'b'],
        'bar': [0, 1],
    })
    cells.remove_unassigned_cells()
    assert len(cells.positions) == n
    assert len(cells.orientations) == n
    assert len(cells.properties) == n


def test_remove_unassigned_3():
    cells = test_module.CellCollection()
    cells.properties = pd.DataFrame({
        'foo': ['a', None],
    })
    cells.remove_unassigned_cells()
    assert cells.positions is None
    assert cells.orientations is None


@pytest.mark.parametrize(
    ("index_offset", "df_index"),
    [
        (0, [0, 1, 2]),
        (1, [1, 2, 3]),
    ]
)
def test_as_dataframe(index_offset, df_index):
    cells = test_module.CellCollection()
    cells.positions = np.random.random((3, 3))
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    df = cells.as_dataframe(index_offset=index_offset)

    assert sorted(df.columns) == ['foo', 'orientation', 'x', 'y', 'z']
    assert_array_equal(df['x'], cells.positions[:, 0])
    assert_array_equal(np.stack(df['orientation']), cells.orientations)
    assert_array_equal(df['foo'].values, cells.properties['foo'].values)

    # check that dataframe is indexed by GIDs
    assert_array_equal(df.index.values, df_index)

    # check that data is copied
    df['foo'] = ['q', 'w', 'v']
    assert_array_equal(cells.properties['foo'], ['a', 'b', 'c'])

    assert df.columns.inferred_type in ('string', 'unicode')


def test_size():
    # Nothing
    cells = test_module.CellCollection()
    assert cells.size() == 0
    assert len(cells) == cells.size()

    # positions only
    cells = test_module.CellCollection()
    cells.positions = np.random.random((3, 3))
    assert cells.size() == 3
    assert len(cells) == cells.size()

    # orientations only
    cells = test_module.CellCollection()
    cells.orientations = random_orientations(3)
    assert cells.size() == 3
    assert len(cells) == cells.size()

    # properties only
    cells = test_module.CellCollection()
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    assert cells.size() == 3
    assert len(cells) == cells.size()

    cells = test_module.CellCollection()
    cells.positions = np.random.random((3, 3))
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    assert cells.size() == 3
    assert len(cells) == cells.size()

    # bad sizes : properties too small
    cells = test_module.CellCollection()
    cells.positions = np.random.random((3, 3))
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b'])
    with pytest.raises(VoxcellError):
        cells.size()


def test_add_properties():
    cells = test_module.CellCollection()
    properties1 = pd.DataFrame({
        'a': [1],
        'b': [2],
    })
    properties2 = pd.DataFrame({
        'b': [3],
        'c': [4],
    })
    combined = pd.DataFrame({
        'a': [1],
        'b': [3],
        'c': [4],
    })

    cells.add_properties(properties1)
    assert_frame_equal(cells.properties, properties1)

    # no duplicates should appear
    cells.add_properties(properties2)
    assert_frame_equal(cells.properties, combined)

    # no overwriting => exception should be raised if column already exists
    with pytest.raises(VoxcellError):
        cells.add_properties(properties1, overwrite=False)


@pytest.mark.parametrize(
    ("index_offset", "df_index"),
    [
        (1, [0, 1]),
        (0, [1, 2]),
    ]
)
def test_from_dataframe_invalid_index(index_offset, df_index):
    df = pd.DataFrame({
        'prop-a': ['a', 'b'],
    }, index=df_index)
    with pytest.raises(VoxcellError, match="Index !="):
        test_module.CellCollection.from_dataframe(df, index_offset=index_offset)


@pytest.mark.parametrize(
    ("index_offset", "df_index"),
    [
        (0, [0, 1]),
        (1, [1, 2]),
    ]
)
def test_from_dataframe_no_positions(index_offset, df_index):
    df = pd.DataFrame({
        'prop-a': ['a', 'b'],
    }, index=df_index)

    cells = test_module.CellCollection.from_dataframe(df, index_offset=index_offset)
    assert cells.positions is None
    assert cells.orientations is None
    assert_frame_equal(cells.properties, df.reset_index(drop=True))


@pytest.mark.parametrize("index_offset", [0, 1])
def test_to_from_dataframe(index_offset):
    cells = test_module.CellCollection()
    cells.positions = random_positions(3)
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    cells.properties['cat'] = pd.Categorical.from_codes(
        codes=np.zeros(3, dtype=np.uint), categories=['a'])

    df = cells.as_dataframe(index_offset=index_offset)
    cells2 = test_module.CellCollection.from_dataframe(df, index_offset=index_offset)
    assert_almost_equal(cells.positions, cells2.positions)
    assert_almost_equal(cells.orientations, cells2.orientations)
    assert_frame_equal(cells.properties, cells2.properties)


def assert_sonata_rotation(filename, good_orientation_type):
    cells = test_module.CellCollection.load_sonata(filename)
    assert cells.orientation_format == good_orientation_type
    check_roundtrip(cells)
    # cannot be included in the roundtrip because the mvd3 conversion will potentially break
    # the format
    with tempcwd():
        original = test_module.CellCollection.load_sonata(filename)
        original.save_sonata("nodes_.h5")
        restored = test_module.CellCollection.load_sonata("nodes_.h5")
        assert original.orientation_format == restored.orientation_format
        for axis in ["x", "y", "z", "w"]:
            assert "rotation_angle_{}axis".format(axis) not in restored.properties
            assert "rotation_angle_{}axis".format(axis) not in original.properties
            assert "orientation_{}".format(axis) not in restored.properties
            assert "orientation_{}".format(axis) not in original.properties


def test_load_sonata_orientations():
    assert_sonata_rotation(SONATA_DATA_PATH / "nodes_eulers.h5", "eulers")
    assert_sonata_rotation(SONATA_DATA_PATH / "nodes_quaternions.h5", "quaternions")
    assert_sonata_rotation(SONATA_DATA_PATH / "nodes_no_rotation.h5", "quaternions")
    with pytest.raises(VoxcellError):
        assert_sonata_rotation(SONATA_DATA_PATH / "nodes_quaternions_w_missing.h5", "quaternions")


def test_set_orientation_type():
    with tempcwd():
        cells = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_eulers.h5")
        assert cells.orientation_format == "eulers"
        cells.orientation_format = "quaternions"
        cells.save_sonata("nodes_.h5")
        restored = test_module.CellCollection.load_sonata("nodes_.h5")
        assert restored.orientation_format == "quaternions"
        for axis in ["x", "y", "z"]:
            assert "rotation_angle_{}axis".format(axis) not in restored.properties

        with pytest.raises(VoxcellError):
            cells.orientation_format = "unknown"


def test_sonata_multipopulation():
    with tempcwd():
        orig = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_eulers.h5")

        orig.population_name = 'A'
        orig.save_sonata("nodes.h5")

        orig.population_name = 'B'
        # this tests that `mode` is used, otherwise the file would be overwritten, and population
        # A wouldn't be accessible below
        orig.save_sonata("nodes.h5", mode='a')

        A = test_module.CellCollection.load_sonata("nodes.h5", population_name="A")
        B = test_module.CellCollection.load_sonata("nodes.h5", population_name="B")
        assert_frame_equal(A.as_dataframe(), B.as_dataframe())
        assert_frame_equal(A.as_dataframe(index_offset=0), B.as_dataframe(index_offset=0))
        assert_frame_equal(A.as_dataframe(index_offset=1), B.as_dataframe(index_offset=1))


def test_check_types():
    # this is a sanity check for the h5py>3.0.0 and the string types
    cells = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_multi_types.h5")
    assert_array_equal(cells.properties["string"],
                       ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG'])
    assert_array_equal(cells.properties["categorical"], ['A', 'A', 'B', 'A', 'A', 'A', 'A'])
    assert_array_equal(cells.properties["int"], [0, 0, 1, 0, 0, 0, 0])
    assert_almost_equal(cells.properties["float"], [0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0])
    assert cells.properties["float"].to_numpy().dtype == np.float32


def test_save_sonata_forced_library():
    with tempcwd():
        orig = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_multi_types.h5")
        orig.save_sonata("nodes_.h5", forced_library=["categorical", "string"])
        with h5py.File("nodes_.h5", "r") as h5:
            assert list(h5['/nodes/default/0/@library'].keys()) == ["categorical", "string"]
        restored = test_module.CellCollection.load_sonata("nodes_.h5")
        assert_equal_cells(orig, restored)


def test_str_and_repr():
    for op in (str, repr):
        cc = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_quaternions.h5")
        ret = op(cc)
        assert 'orientation_x' in ret
        assert 'orientation_y' in ret
        assert 'orientation_z' in ret
        assert 'orientation_w' in ret

        cc = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_eulers.h5")
        ret = op(cc)
        assert 'rotation_angle_xaxis' in ret
        assert 'rotation_angle_yaxis' in ret
        assert 'rotation_angle_zaxis' in ret

        cc = test_module.CellCollection.load_sonata(SONATA_DATA_PATH / "nodes_multi_types.h5")
        ret = op(cc)
        assert 'categorical' in ret
        assert 'float' in ret
        assert 'string' in ret
        assert 'int' in ret
