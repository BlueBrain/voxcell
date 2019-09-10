import os
import shutil
import tempfile

from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

import nose.tools as nt

import mock

from voxcell import VoxcellError
from voxcell.math_utils import euler2mat

import voxcell.sonata.node_population as test_module


@nt.raises(VoxcellError)
def test_open_population_fail():
    storage = mock.Mock()
    storage.population_names = ['a', 'b']
    with mock.patch(test_module.__name__ + '.NodeStorage') as NodeStorage:
        test_module._open_population(mock.ANY)


def random_positions(n):
    return np.random.random((n, 3))


def random_angles(n):
    return 2 * np.pi * np.random.random(n)


def random_orientations(n):
    return euler2mat(
        random_angles(n),
        random_angles(n),
        random_angles(n)
    )


def test_attributes_1():
    n = 2
    cells = test_module.NodePopulation('test', n)
    cells.attributes['foo'] = np.array([42, 43])
    cells.attributes['bar'] = ['a', 'b']
    pdt.assert_frame_equal(
        cells.attributes,
        pd.DataFrame({
            'foo': [42, 43],
            'bar': ['a', 'b'],
        }),
        check_like=True
    )


@nt.raises(VoxcellError)
def test_attributes_2():
    n = 1
    cells = test_module.NodePopulation('test', n)
    cells.attributes['foo'] = [42]
    cells.attributes['foo'] = [43]  # attempt to overwrite


@nt.raises(VoxcellError)
def test_attributes_3():
    n = 1
    cells = test_module.NodePopulation('test', n)
    cells.attributes['@foo'] = [42]  # reserved attribute names


@nt.raises(VoxcellError)
def test_attributes_4():
    n = 1
    cells = test_module.NodePopulation('test', n)
    cells.attributes['dynamics_params'] = [42]  # reserved group name


@nt.raises(ValueError)
def test_attributes_5():
    n = 1
    cells = test_module.NodePopulation('test', n)
    cells.attributes['foo'] = [42, 43]  # invalid length


def test_dynamics_attributes():
    n = 2
    cells = test_module.NodePopulation('test', n)
    cells.dynamics_attributes['foo'] = np.array([42, 43])
    cells.dynamics_attributes['bar'] = ['a', 'b']
    pdt.assert_frame_equal(
        cells.dynamics_attributes,
        pd.DataFrame({
            'foo': [42, 43],
            'bar': ['a', 'b'],
        }),
        check_like=True
    )
    nt.assert_true(cells.attributes.empty)


def test_positions():
    n = 2
    cells = test_module.NodePopulation('test', n)
    cells.attributes['foo'] = ['a', 'b']
    positions = random_positions(n)
    cells.positions = positions
    pdt.assert_frame_equal(
        cells.attributes,
        pd.DataFrame({
            'x': positions[:, 0],
            'y': positions[:, 1],
            'z': positions[:, 2],
            'foo': ['a', 'b'],
        }),
        check_like=True
    )
    nt.assert_is_instance(cells.positions, np.ndarray)
    npt.assert_array_equal(cells.positions, positions)


def test_orientations_1():
    n = 2
    ax = random_angles(n)
    az = random_angles(n)
    orientations = euler2mat(az, np.zeros(n), ax)
    cells = test_module.NodePopulation('test', n)
    cells.attributes['rotation_angle_zaxis'] = az
    cells.attributes['rotation_angle_xaxis'] = ax
    # 'rotation_angle_yaxis' not assigned => implicit value '0'
    nt.assert_is_instance(cells.orientations, np.ndarray)
    npt.assert_almost_equal(cells.orientations, orientations)


def test_orientations_2():
    n = 2
    orientations = random_orientations(n)
    cells = test_module.NodePopulation('test', n)
    cells.orientations = orientations
    nt.assert_is_instance(cells.orientations, np.ndarray)
    npt.assert_almost_equal(cells.orientations, orientations)


def test_to_dataframe():
    n = 2
    cells = test_module.NodePopulation('test', n)
    cells.attributes['foo'] = np.array(['a', 'b'])
    cells.dynamics_attributes['bar'] = np.array(['c', 'd'])
    actual = cells.to_dataframe()
    pdt.assert_frame_equal(
        actual,
        pd.DataFrame({
            'foo': ['a', 'b'],
            '@dynamics:bar': ['c', 'd'],
        }),
        check_like=True
    )
    # check that data is copied
    actual['foo'] = ['e', 'f']
    npt.assert_equal(cells.attributes['foo'].values, ['a', 'b'])


@contextmanager
def tempcwd():
    cwd = os.getcwd()
    dirname = tempfile.mkdtemp()
    os.chdir(dirname)
    try:
        yield dirname
    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def assert_equal_cells(c0, c1):
    pdt.assert_frame_equal(
        c0.attributes,
        c1.attributes,
        check_like=True,
        check_names=True,
    )
    pdt.assert_frame_equal(
        c0.dynamics_attributes,
        c1.dynamics_attributes,
        check_like=True,
        check_names=True,
    )


def check_roundtrip(original, library_properties=None):
    with tempcwd() as tmp:
        path = os.path.join(tmp, 'nodes.h5')
        original.save(path, library_properties)
        restored = test_module.NodePopulation.load(path)
        assert_equal_cells(original, restored)
        return restored


def test_roundtrip_1():
    n = 10
    cells = test_module.NodePopulation('test', n)
    cells.positions = random_positions(n)
    cells.orientations = random_orientations(n)
    cells.attributes['mtype'] = np.random.choice(['L5_NGC', 'L5_BTC', 'L6_LBC'], n)
    cells.attributes['etype'] = np.random.choice(['cADpyr', 'dNAC', 'bSTUT'], n)
    cells.dynamics_attributes['threshold'] = np.random.random(n)
    check_roundtrip(cells)

    #with some columns turned in to @library enum style
    check_roundtrip(cells, ['mtype', 'etype', ])


def test_roundtrip_2():
    cells = test_module.NodePopulation('test', 1)
    check_roundtrip(cells)
