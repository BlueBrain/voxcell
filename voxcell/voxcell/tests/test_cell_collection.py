import tempfile
import os
import shutil
import h5py
from contextlib import contextmanager
from nose.tools import eq_, assert_raises

from voxcell import core, VoxcellError
from numpy.testing import assert_equal, assert_almost_equal
from pandas.util.testing import assert_frame_equal
import numpy as np
import pandas as pd


def euler_to_matrix(bank, attitude, heading):
    '''build 3x3 rotation matrices from arrays of euler angles

    Based on algorigthm described here:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm

    Args:
        bank: rotation around X
        attitude: rotation around Z
        heading: rotation around Y
    '''

    sa = np.sin(attitude)
    ca = np.cos(attitude)
    sb = np.sin(bank)
    cb = np.cos(bank)
    sh = np.sin(heading)
    ch = np.cos(heading)

    m = np.vstack([ch*ca, -ch*sa*cb + sh*sb, ch*sa*sb + sh*cb,
                   sa, ca*cb, -ca*sb,
                   -sh*ca, sh*sa*cb + ch*sb, -sh*sa*sb + ch*cb]).transpose()

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


@contextmanager
def tempcwd():
    '''timer contextmanager'''
    cwd = os.getcwd()
    dirname = tempfile.mkdtemp(prefix='bbtests_')
    os.chdir(dirname)
    print 'working dir:', dirname

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
    assert_frame_equal(c0.properties.sort(axis=1), c1.properties.sort(axis=1), check_names=True)


def check_roundtrip(original):
    with tempcwd():
        original.save('cells.h5')
        restored = core.CellCollection.load('cells.h5')
        assert_equal_cells(original, restored)
        return restored


def test_roundtrip_empty():
    cells = core.CellCollection()
    check_roundtrip(cells)


def test_roundtrip_properties_numeric_single():
    cells = core.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75]
    check_roundtrip(cells)


def test_roundtrip_properties_numeric_multiple():
    cells = core.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75, 0]
    cells.properties['z-factor'] = [0, 0.75, 0.5, 0.25]
    check_roundtrip(cells)


def test_roundtrip_properties_text_single():
    cells = core.CellCollection()
    cells.properties['y-type'] = ['pretty', 'ugly', 'pretty']
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()


def test_roundtrip_properties_text_multiple():
    cells = core.CellCollection()
    cells.properties['y-type'] = ['pretty', 'ugly', 'ugly', 'pretty']
    cells.properties['z-type'] = ['red', 'blue', 'green', 'alpha']
    restored = check_roundtrip(cells)
    restored.properties['y-type'].to_frame()
    restored.properties['z-type'].to_frame()


def test_roundtrip_positions():
    cells = core.CellCollection()
    cells.positions = np.random.random((10, 3))
    check_roundtrip(cells)


def test_roundtrip_orientations():
    cells = core.CellCollection()
    cells.orientations = random_orientations(10)
    check_roundtrip(cells)


def test_roundtrip_complex():
    cells = core.CellCollection()
    n = 10

    cells.positions = np.random.random((n, 3))
    cells.orientations = random_orientations(n)
    cells.properties['synapse_class'] = np.random.choice(['INH', 'EXC'], n)
    cells.properties['mtype'] = np.random.choice(['L5_NGC', 'L5_BTC', 'L6_LBC'], n)
    cells.properties['etype'] = np.random.choice(['cADpyr', 'dNAC', 'bSTUT'], n)
    cells.properties['morphology'] = np.random.choice([
        'dend-C250500A-P3_axon-C190898A-P2_-_Scale_x1.000_y1.025_z1.000_-_Clone_2',
        'C240300C1_-_Scale_x1.000_y0.975_z1.000_-_Clone_55',
        'dend-Fluo15_right_axon-Fluo2_right_-_Clone_37'
    ], n)

    check_roundtrip(cells)


def test_remove_unassigned():
    ''' test that cells with unassigned properties are removed '''
    cells = core.CellCollection()
    n = 5

    cells.positions = np.random.random((n, 3))
    cells.orientations = random_orientations(n)
    cells.properties['foo'] = np.array(['', 'a', None, 'b', 'c'])
    cells.properties['bar'] = np.array([0, None, 2, 3, 4])
    cells.remove_unassigned_cells()
    eq_(len(cells.positions), 3)
    eq_(len(cells.orientations), 3)
    assert_equal(cells.properties['foo'], np.array(['', 'b', 'c']))
    assert_equal(cells.properties['bar'], np.array([0, 3, 4]))


def test_circuit_seeds():
    ''' test that 4 seeds are generated in the MVD3 during save '''
    cells = core.CellCollection()
    cells.properties['foo'] = ['a', 'b', 'c']
    with tempcwd():
        cells.save('cells.h5')
        with h5py.File('cells.h5') as f:
            eq_(f['circuit']['seeds'].shape, (4,))


def test_as_dataframe():
    cells = core.CellCollection()
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


def test_add_properties():
    cells = core.CellCollection()
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
