from brainbuilder.utils import genbrain as gb

from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal

import numpy as np
import tempfile
import os
import shutil
from contextlib import contextmanager


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
    assert_equal(c0.orientations, c1.orientations)
    assert_frame_equal(c0.properties.sort(axis=1), c1.properties.sort(axis=1), check_names=True)


def check_roundtrip(original):
    with tempcwd():
        original.serialize('cells.h5')
        restored = gb.CellCollection.deserialize('cells.h5')
        assert_equal_cells(original, restored)


def test_roundtrip_empty():
    cells = gb.CellCollection()
    check_roundtrip(cells)


def test_roundtrip_properties_numeric_single():
    cells = gb.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75]
    check_roundtrip(cells)


def test_roundtrip_properties_numeric_multiple():
    cells = gb.CellCollection()
    cells.properties['y-factor'] = [0.25, 0.5, 0.75, 0]
    cells.properties['z-factor'] = [0, 0.75, 0.5, 0.25]
    check_roundtrip(cells)


def test_roundtrip_properties_text_single():
    cells = gb.CellCollection()
    cells.properties['y-type'] = ['pretty', 'ugly', 'pretty']
    check_roundtrip(cells)


def test_roundtrip_properties_text_multiple():
    cells = gb.CellCollection()
    cells.properties['y-type'] = ['pretty', 'ugly', 'ugly', 'pretty']
    cells.properties['z-type'] = ['red', 'blue', 'green', 'alpha']
    check_roundtrip(cells)


def test_roundtrip_positions():
    cells = gb.CellCollection()
    cells.positions = np.random.random((10, 3))
    check_roundtrip(cells)


def test_roundtrip_orientations():
    cells = gb.CellCollection()
    cells.orientations = np.random.random((10, 3, 3))
    check_roundtrip(cells)


def test_roundtrip_complex():
    cells = gb.CellCollection()
    n = 10

    cells.positions = np.random.random((n, 3))
    cells.orientations = np.random.random((n, 3, 3))
    cells.properties['synapse_class'] = np.random.choice(['inhibitory', 'excitatory'], n)
    cells.properties['mtype'] = np.random.choice(['L5_NGC', 'L5_BTC', 'L6_LBC'], n)
    cells.properties['etype'] = np.random.choice(['cADpyr', 'dNAC', 'bSTUT'], n)
    cells.properties['morphology'] = np.random.choice([
        'dend-C250500A-P3_axon-C190898A-P2_-_Scale_x1.000_y1.025_z1.000_-_Clone_2',
        'C240300C1_-_Scale_x1.000_y0.975_z1.000_-_Clone_55',
        'dend-Fluo15_right_axon-Fluo2_right_-_Clone_37'
    ], n)

    check_roundtrip(cells)
