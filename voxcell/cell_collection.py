""" Cell collection access / writer. """

import h5py
import numpy as np
import pandas as pd

from six import iteritems, text_type

from voxcell.exceptions import VoxcellError
from voxcell.quaternion import matrices_to_quaternions, quaternions_to_matrices


class CellCollection(object):
    '''Encapsulates all the data related to a collection of cells that compose a circuit.

    Multi-dimensional properties (such as positions and orientations) are attributes.
    General properties are a in a pandas DataFrame object "properties".
    '''
    def __init__(self):
        self.positions = None
        self.orientations = None
        self.properties = pd.DataFrame()
        self.meta = {}

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
        '''save this cell collection to HDF5

        Args:
            filename(str): fullpath to filename to write
        '''
        with h5py.File(filename, 'w') as f:
            f.attrs.update(self.meta)

            f.create_group('cells')
            f.create_group('library')

            if self.positions is not None:
                f.create_dataset('cells/positions', data=self.positions)

            if self.orientations is not None:
                f.create_dataset('cells/orientations',
                                 data=matrices_to_quaternions(self.orientations))

            # TODO this should be managed by the application that requires that.
            # This is in the current MVD3 spec and this is a legacy from MVD2.
            # This is not used in load
            f.create_dataset('circuit/seeds',
                             data=np.random.random_sample((4,)).astype(np.float64))

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
        '''load a cell collection from HDF5

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        '''

        cells = cls()

        with h5py.File(filename, 'r') as f:
            cells.meta.update(f.attrs)

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
