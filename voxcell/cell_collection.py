""" Cell collection access / writer. """

import h5py
import numpy as np
import pandas as pd
import quaternion as quat

from six import iteritems, text_type

from voxcell.exceptions import VoxcellError
from voxcell.utils import deprecate


def matrices_to_quaternions(m):
    """
    Build quaternions from an array of 3x3 rotation matrices

    Args:
        m: A Nx3x3 numpy array containing N rotation matrices.

    Returns:
        Nx4 numpy array containing a unit quaternion for each rotation matrix.
        The quaternion components are stored as (x, y, z, w)
    """
    q = quat.as_float_array(quat.from_rotation_matrix(m, nonorthogonal=False))
    # change quaternion component order: (w, x, y, z) -> (x, y, z, w)
    return np.roll(q, -1, axis=-1)


def quaternions_to_matrices(q):
    """
    Build 3x3 rotation matrices from an array of quaternions.

    Args:
        q: A Nx4 numpy array containing a quaternion for each rotation matrix.
        The quaternion components are stored as (x, y, z, w)

    Returns:
        A Nx3x3 numpy array containing N rotation matrices.
    """
    # change quaternion component order: (x, y, z, w) -> (w, x, y, z)
    q = np.roll(q, 1, axis=-1)
    return quat.as_rotation_matrix(quat.from_float_array(q))


class CellCollection(object):
    '''Encapsulates all the data related to a collection of cells that compose a circuit.

    Multi-dimensional properties (such as positions and orientations) are attributes.
    General properties are a in a pandas DataFrame object "properties".
    '''
    def __init__(self):
        self.positions = None
        self.orientations = None
        self.properties = pd.DataFrame()
        self.seeds = np.random.random(4)  # used by Functionalizer later (sic!)

    def add_properties(self, new_properties, overwrite=True):
        '''adds new columns to the properties DataFrame

        Args:
            new_properties: a pandas DataFrame object
            overwrite: if True, overwrites columns with the same name.
            Otherwise, a VoxcellError is raised.
        '''
        for name, prop in new_properties.iteritems():
            if (not overwrite) and (name in self.properties):
                raise VoxcellError("Column '{0}' already exists".format(name))
            self.properties[name] = prop

    def remove_unassigned_cells(self):
        ''' remove cells with one or more unassigned property '''
        idx_unassigned = self.properties[self.properties.isnull().any(axis=1)].index
        self.properties = self.properties.drop(idx_unassigned)
        self.orientations = np.delete(self.orientations, idx_unassigned, 0)
        self.positions = np.delete(self.positions, idx_unassigned, 0)

    def as_dataframe(self):
        ''' return a dataframe with all cell properties '''
        result = self.properties.copy()
        if self.positions is not None:
            result['x'] = self.positions[:, 0]
            result['y'] = self.positions[:, 1]
            result['z'] = self.positions[:, 2]
        if self.orientations is not None:
            result['orientation'] = [m for m in self.orientations]
        result.index = 1 + np.arange(len(result))
        return result

    @classmethod
    def from_dataframe(cls, df):
        ''' return a CellCollection object from a dataframe of cell properties '''
        if not (df.index == 1 + np.arange(len(df))).all():
            raise VoxcellError("Index != 1..{0} (got: {1})".format(len(df), df.index.values))
        result = cls()
        if 'x' in df:
            result.positions = df[['x', 'y', 'z']].values
        if 'orientation' in df:
            result.orientations = np.stack(df['orientation'])
        props = set(df.columns) - set(['x', 'y', 'z', 'orientation'])
        result.properties = df[list(props)].reset_index(drop=True)
        return result

    def save(self, filename):
        # pylint: disable=missing-docstring
        deprecate.warn(
            "CellCollection.save() is deprecated, please use CellCollection.save_mvd3() instead"
        )
        self.save_mvd3(filename)

    def save_mvd3(self, filename):
        '''save this cell collection to HDF5

        Args:
            filename(str): fullpath to filename to write
        '''
        with h5py.File(filename, 'w') as f:
            f.create_group('cells')
            f.create_group('library')

            if self.positions is not None:
                f.create_dataset('cells/positions', data=self.positions)

            if self.orientations is not None:
                f.create_dataset('cells/orientations',
                                 data=matrices_to_quaternions(self.orientations))

            # TODO this should be managed by the application that requires that.
            # This is in the current MVD3 spec and this is a legacy from MVD2.
            if self.seeds is not None:
                f.create_dataset('circuit/seeds', data=self.seeds)

            for name, series in self.properties.iteritems():
                data = series.values

                if data.dtype == np.object:
                    # numpy uses "np.object" to represent variable size strings
                    # however, h5py doesn't like this.
                    # http://docs.h5py.org/en/latest/strings.html

                    unique_values, indices = np.unique(data, return_inverse=True)
                    f.create_dataset('cells/properties/' + name, data=indices.astype(np.uint32))

                    dt = h5py.special_dtype(vlen=text_type)
                    f.create_dataset('library/' + name, data=unique_values, dtype=dt)

                else:
                    f.create_dataset('cells/properties/' + name, data=data)

    @classmethod
    def load(cls, filename):
        # pylint: disable=missing-docstring
        deprecate.warn(
            "CellCollection.load() is deprecated, please use CellCollection.load_mvd3() instead"
        )
        return cls.load_mvd3(filename)

    @classmethod
    def load_mvd3(cls, filename):
        '''load a cell collection from HDF5

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        '''

        cells = cls()

        with h5py.File(filename, 'r') as f:
            if 'circuit/seeds' in f:
                cells.seeds = np.array(f['circuit/seeds'])

            data = f['cells']
            if 'positions' in data:
                cells.positions = np.array(data['positions'])

            if 'orientations' in data:
                cells.orientations = np.array(data['orientations'])
                cells.orientations = quaternions_to_matrices(cells.orientations)

            if 'properties' in data:
                properties = data['properties']

                for name, data in iteritems(properties):
                    data = np.array(data)

                    if name in f['library']:
                        labels = f['library'][name]
                        # older versions of h5py don't properly return
                        # variable length strings that pandas can consume
                        # (ie: they fail the to_frame() command with a KeyError
                        # due to the vlen in the dtype, force the conversion
                        # using the numpy array
                        if labels.dtype.names and 'vlen' in labels.dtype.names:
                            unique_values = np.array(labels, dtype=object)
                        else:
                            unique_values = np.array(labels)
                        cells.properties[name] = unique_values[data]

                    else:
                        cells.properties[name] = data

        return cells
