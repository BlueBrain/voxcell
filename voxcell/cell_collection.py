""" Cell collection access / writer. """

import h5py
import numpy as np
import pandas as pd

from six import iteritems, text_type

from voxcell.exceptions import VoxcellError
from voxcell.quaternion import matrices_to_quaternions, quaternions_to_matrices


def _load_property(properties, name, values, library_group=None):
    """Loads single property with respect to a library group if presented.

    Args:
        properties (pd.DataFrame): properties
        name (str): property name
        values (array-like): property values
        library_group (h5py.Group): library group
    """
    values = np.array(values)
    if library_group is not None and name in library_group:
        labels = library_group[name]
        # older versions of h5py don't properly return
        # variable length strings that pandas can consume
        # (ie: they fail the to_frame() command with a KeyError
        # due to the vlen in the dtype, force the conversion
        # using the numpy array
        if labels.dtype.names and 'vlen' in labels.dtype.names:
            unique_values = np.array(labels, dtype=object)
        else:
            unique_values = np.array(labels)
        properties[name] = unique_values[values]
    else:
        properties[name] = values


class CellCollection(object):
    '''Encapsulates all the data related to a collection of cells that compose a circuit.

    Multi-dimensional properties (such as positions and orientations) are attributes.
    General properties are a in a pandas DataFrame object "properties".
    '''

    # properties that start with it are dynamic, and handled appropriately, see `dynamics_params` in
    # https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-nodes
    SONATA_DYNAMIC_PROPERTY = '@dynamics:'

    def __init__(self, population_name='default'):
        # SONATA population name, currently assume a single population collection
        self.population_name = population_name
        self.positions = None
        self.orientations = None
        self.properties = pd.DataFrame()

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
        self.properties.reset_index(inplace=True, drop=True)
        if self.orientations is not None:
            self.orientations = np.delete(self.orientations, idx_unassigned, 0)
        if self.positions is not None:
            self.positions = np.delete(self.positions, idx_unassigned, 0)

    def as_dataframe(self):
        ''' return a dataframe with all cell properties '''
        result = self.properties.copy()
        if self.positions is not None:
            result['x'] = self.positions[:, 0]
            result['y'] = self.positions[:, 1]
            result['z'] = self.positions[:, 2]
        if self.orientations is not None:
            result['orientation'] = list(self.orientations)

        result.index = 1 + np.arange(len(result))

        result.columns = map(str, result.columns)

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
        """Saves this cell collection to HDF5 file in MVD3 or SONATA format.

        Args:
            filename: filepath to write. If it ends with '.mvd3' then it is treated as MVD3,
                otherwise as SONATA.
        """
        if str(filename).lower().endswith('mvd3'):
            self.save_mvd3(filename)
        else:
            self.save_sonata(filename)

    def save_mvd3(self, filename):
        '''save this cell collection to mvd3 HDF5

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
        """Loads CellCollection from a file.

        Args:
            filename: filepath to cells file. If it ends with '.mvd3' then it is treated as
                MVD3 circuit, otherwise as SONATA.

        Returns:
            CellCollection: loaded cells
        """
        if str(filename).lower().endswith('mvd3'):
            return cls.load_mvd3(filename)
        return cls.load_sonata(filename)

    @classmethod
    def load_mvd3(cls, filename):
        '''load a cell collection from mvd3 HDF5

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        '''

        cells = cls()

        with h5py.File(filename, 'r') as f:
            data = f['cells']
            if 'positions' in data:
                cells.positions = np.array(data['positions'])

            if 'orientations' in data:
                cells.orientations = np.array(data['orientations'])
                cells.orientations = quaternions_to_matrices(cells.orientations)

            if 'properties' in data:
                for name, values in iteritems(data['properties']):
                    _load_property(cells.properties, name, values, f.get('library'))

        return cells

    def save_sonata(self, filename):
        """Save this cell collection to sonata HDF5.

        Args:
            filename(str): fullpath to filename to write
        """
        with h5py.File(filename, 'w') as h5f:
            population = h5f.create_group('/nodes/%s' % self.population_name)
            population.create_dataset('node_type_id', data=np.full(len(self.properties), -1))
            group = population.create_group('0')
            str_dt = h5py.special_dtype(vlen=text_type)
            for name, series in self.properties.iteritems():
                values = series.values
                if name.startswith(self.SONATA_DYNAMIC_PROPERTY):
                    name = name.split(self.SONATA_DYNAMIC_PROPERTY)[1]
                    dt = str_dt if series.dtype == np.object else series.dtype
                    group.create_dataset('dynamics_params/' + name, data=values, dtype=dt)
                elif series.dtype == np.object:
                    unique_values, indices = np.unique(values, return_inverse=True)
                    if len(unique_values) < len(values):
                        group.create_dataset(name, data=indices.astype(np.uint32))
                        group.create_dataset('@library/' + name, data=unique_values, dtype=str_dt)
                    else:
                        group.create_dataset(name, data=values, dtype=str_dt)
                else:
                    group.create_dataset(name, data=values)

            if self.orientations is not None:
                quaternions = matrices_to_quaternions(self.orientations)
                group.create_dataset('orientation_x', data=quaternions[:, 0])
                group.create_dataset('orientation_y', data=quaternions[:, 1])
                group.create_dataset('orientation_z', data=quaternions[:, 2])
                group.create_dataset('orientation_w', data=quaternions[:, 3])
            if self.positions is not None:
                group.create_dataset('x', data=self.positions[:, 0])
                group.create_dataset('y', data=self.positions[:, 1])
                group.create_dataset('z', data=self.positions[:, 2])

    @classmethod
    def load_sonata(cls, filename):
        """Loads a cell collection from sonata HDF5.

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        """
        cells = cls()

        with h5py.File(filename, 'r') as h5f:
            population_names = list(h5f['/nodes'].keys())
            assert len(population_names) == 1, 'Single population is supported only'
            cells.population_name = population_names[0]
            population = h5f['/nodes/' + population_names[0]]
            assert '0' in population, 'Single group "0" is supported only'
            group = population['0']

            if 'x' in group:
                cells.positions = np.vstack((group['x'], group['y'], group['z'])).T
            if 'orientation_x' in group:
                quaternions = np.vstack((group['orientation_x'], group['orientation_y'],
                                         group['orientation_z'], group['orientation_w']))
                cells.orientations = quaternions_to_matrices(quaternions.T)
            properties_names = set(group.keys()) - {'x', 'y', 'z', 'orientation_x', 'orientation_y',
                                                    'orientation_z', 'orientation_w'}
            for name in properties_names:
                if not isinstance(group[name], h5py.Dataset):
                    continue
                _load_property(cells.properties, name, group[name], group.get('@library'))

            if 'dynamics_params' in group:
                for name, values in iteritems(group['dynamics_params']):
                    if not isinstance(values, h5py.Dataset):
                        continue
                    cells.properties[cls.SONATA_DYNAMIC_PROPERTY + name] = np.array(values)

        return cells
