import tempfile
import os
import shutil
from contextlib import contextmanager
from nose.tools import ok_, eq_, assert_raises, assert_is_none, raises

import voxcell.cell_collection as test_module
from voxcell import VoxcellError

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal
import pandas as pd
from pandas.api.types import CategoricalDtype, is_categorical_dtype as is_cat
from pandas.testing import assert_frame_equal, assert_series_equal

SONATA_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/sonata/')


def euler_to_matrix(bank, attitude, heading):
    '''build 3x3 rotation matrices from arrays of euler angles

    Based on algorigthm described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm

    Args:
        bank: rotation around X
        attitude: rotation around Z
        heading: rotation around Y
    '''

    sa, ca = np.sin(attitude), np.cos(attitude)
    sb, cb = np.sin(bank), np.cos(bank)
    sh, ch = np.sin(heading), np.cos(heading)

    m = np.vstack([ch * ca, -ch * sa * cb + sh * sb, ch * sa * sb + sh * cb,
                   sa, ca * cb, -ca * sb,
                   -sh * ca, sh * sa * cb + ch * sb, -sh * sa * sb + ch * cb]).transpose()

    return m.reshape(m.shape[:-1] + (3, 3))


def test_euler_to_matrix():  # testing the test
    n = 2

    assert_equal(
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
    '''timer contextmanager'''
    cwd = os.getcwd()
    dirname = tempfile.mkdtemp(prefix='bbtests_')
    os.chdir(dirname)
    try:
        yield dirname
    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def assert_equal_cells(c0, c1):
    assert_equal(c0.positions, c1.positions)
    if c0.orientations is None:
        eq_(c0.orientations, c1.orientations)
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


def test_roundtrip_empty():
    cells = test_module.CellCollection()
    check_roundtrip(cells)


def test_roundtrip_none():
    cells = test_module.CellCollection()
    cells.properties['y-factor'] = [0.25, np.nan, 0.75]
    assert_raises(VoxcellError, check_roundtrip, cells)
    cells.properties['y-factor'] = [None, 0.1, 0.75]
    assert_raises(VoxcellError, check_roundtrip, cells)


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
    assert_equal(cells.properties['y-type'].dtypes, np.object)
    assert_equal(cells.properties['z-type'].dtypes, np.object)
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()
    # y-type should be categorical now
    assert is_cat(restored.properties['y-type'])
    assert_equal(cells.properties['z-type'].dtypes, np.object)
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
    assert_equal(cells.positions, positions[[0, 3, 4]])
    assert_equal(cells.orientations, orientations[[0, 3, 4]])
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
    assert_equal(len(cells.positions), n)
    assert_equal(len(cells.orientations), n)
    assert_equal(len(cells.properties), n)


def test_remove_unassigned_3():
    cells = test_module.CellCollection()
    cells.properties = pd.DataFrame({
        'foo': ['a', None],
    })
    cells.remove_unassigned_cells()
    assert_is_none(cells.positions)
    assert_is_none(cells.orientations)


def test_as_dataframe():
    cells = test_module.CellCollection()
    cells.positions = np.random.random((3, 3))
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    df = cells.as_dataframe()
    assert_equal(sorted(df.columns), ['foo', 'orientation', 'x', 'y', 'z'])
    assert_equal(df['x'], cells.positions[:, 0])
    assert_equal(np.stack(df['orientation']), cells.orientations)
    assert_equal(df['foo'].values, cells.properties['foo'].values)

    # check that dataframe is indexed by GIDs
    assert_equal(df.index.values, [1, 2, 3])

    # check that data is copied
    df['foo'] = ['q', 'w', 'v']
    assert_equal(cells.properties['foo'].values, ['a', 'b', 'c'])

    ok_(df.columns.inferred_type in ('string', 'unicode'))


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
    assert_raises(
        VoxcellError,
        cells.add_properties, properties1, overwrite=False
    )


@raises(VoxcellError)
def test_from_dataframe_invalid_index():
    df = pd.DataFrame({
        'prop-a': ['a', 'b'],
    })
    test_module.CellCollection.from_dataframe(df)


def test_from_dataframe_no_positions():
    df = pd.DataFrame({
        'prop-a': ['a', 'b'],
    }, index=[1, 2])

    cells = test_module.CellCollection.from_dataframe(df)
    assert_is_none(cells.positions)
    assert_is_none(cells.orientations)
    assert_frame_equal(cells.properties, df.reset_index(drop=True))


def test_to_from_dataframe():
    cells = test_module.CellCollection()
    cells.positions = random_positions(3)
    cells.orientations = random_orientations(3)
    cells.properties['foo'] = np.array(['a', 'b', 'c'])
    cells.properties['cat'] = pd.Categorical.from_codes(
        codes=np.zeros(3, dtype=np.uint), categories=['a'])

    cells2 = test_module.CellCollection.from_dataframe(cells.as_dataframe())
    assert_almost_equal(cells.positions, cells2.positions)
    assert_almost_equal(cells.orientations, cells2.orientations)
    assert_frame_equal(cells.properties, cells2.properties)


def assert_sonata_rotation(filename, good_orientation_type):
    cells = test_module.CellCollection.load_sonata(filename)
    assert_equal(cells.orientation_format, good_orientation_type)
    check_roundtrip(cells)
    # cannot be included in the roundtrip because the mvd3 conversion will potentially break
    # the format
    with tempcwd():
        original = test_module.CellCollection.load_sonata(filename)
        original.save_sonata("nodes_.h5")
        restored = test_module.CellCollection.load_sonata("nodes_.h5")
        assert_equal(original.orientation_format, restored.orientation_format)
        for axis in ["x", "y", "z", "w"]:
            assert "rotation_angle_{}axis".format(axis) not in restored.properties
            assert "rotation_angle_{}axis".format(axis) not in original.properties
            assert "orientation_{}".format(axis) not in restored.properties
            assert "orientation_{}".format(axis) not in original.properties


def test_load_sonata_orientations():
    assert_sonata_rotation(os.path.join(SONATA_DATA_PATH, "nodes_eulers.h5"), "eulers")
    assert_sonata_rotation(os.path.join(SONATA_DATA_PATH, "nodes_quaternions.h5"), "quaternions")
    assert_sonata_rotation(os.path.join(SONATA_DATA_PATH, "nodes_no_rotation.h5"), "quaternions")
    with assert_raises(VoxcellError):
        assert_sonata_rotation(os.path.join(SONATA_DATA_PATH, "nodes_quaternions_w_missing.h5"), "quaternions")


def test_set_orientation_type():
    with tempcwd():
        cells = test_module.CellCollection.load_sonata(os.path.join(SONATA_DATA_PATH, "nodes_eulers.h5"))
        assert_equal(cells.orientation_format, "eulers")
        cells.orientation_format = "quaternions"
        cells.save_sonata("nodes_.h5")
        restored = test_module.CellCollection.load_sonata("nodes_.h5")
        assert_equal(restored.orientation_format, "quaternions")
        for axis in ["x", "y", "z"]:
            assert "rotation_angle_{}axis".format(axis) not in restored.properties

        with assert_raises(VoxcellError):
            cells.orientation_format = "unknown"
