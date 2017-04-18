'''core container classes'''

import itertools
import copy
import json
import logging

import h5py
import numpy as np
import pandas as pd
import nrrd

from six import iteritems, text_type

from voxcell import math_utils, VoxcellError
from voxcell.quaternion import matrices_to_quaternions, quaternions_to_matrices

L = logging.getLogger(__name__)


def _sub_dict(dict_, keys):
    """ Get a dictionary for subset of keys. """
    return {k: dict_[k] for k in keys}


class VoxelData(object):
    '''wrap volumetric data and some basic metadata'''

    OUT_OF_BOUNDS = -1

    def __init__(self, raw, voxel_dimensions, offset=None):
        '''
        Note that he units for the metadata will depend on the atlas being used.

        Args:
            raw(numpy.ndarray): actual voxel values
            voxel_dimensions(tuple of numbers): size of each voxel in space.
            offset(tuple of numbers): offset from an external atlas origin
        '''
        voxel_dimensions = np.array(voxel_dimensions, dtype=np.float32)
        if len(voxel_dimensions.shape) > 1:
            raise VoxcellError("voxel_dimensions should be a 1-d array (got: {0})".format(
                len(voxel_dimensions.shape)
            ))

        self.voxel_dimensions = voxel_dimensions
        n_dim = len(self.voxel_dimensions)

        if offset is None:
            self.offset = np.zeros(n_dim)
        else:
            offset = np.array(offset, dtype=np.float32)
            if offset.shape != (n_dim,):
                raise VoxcellError("'offset' shape should be: {0} (got: {1})".format(
                    (n_dim,), offset.shape
                ))
            self.offset = offset

        if len(raw.shape) < n_dim:
            raise VoxcellError("'raw' should have at least {0} dimensions (got: {1})".format(
                n_dim, len(raw.shape)
            ))
        self.raw = raw

    @classmethod
    def load_nrrd(cls, nrrd_path):
        ''' read volumetric data from a nrrd file '''
        raw, option = nrrd.read(nrrd_path)

        if 'space directions' in option:
            directions = np.array(option['space directions'], dtype=np.float32)
            if not math_utils.is_diagonal(directions):
                raise NotImplementedError("Only diagonal space directions supported at the moment")
            spacings = directions.diagonal()
        elif 'spacings' in option:
            spacings = np.array(option['spacings'], dtype=np.float32)
        else:
            raise VoxcellError("spacings not defined in nrrd")

        offset = None
        if 'space origin' in option:
            offset = tuple(option['space origin'])

        return cls(raw, spacings, offset)

    def save_nrrd(self, nrrd_path):
        '''save a VoxelData to an nrrd file

        Args:
            nrrd_path(string): full path to nrrd file
        '''
        #from: http://teem.sourceforge.net/nrrd/format.html#space
        options = {
            'spacings': self.voxel_dimensions,
            'space origin': self.offset,
        }
        nrrd.write(nrrd_path, self.raw, options=options)

    def lookup(self, positions, outer_value=None):
        '''find the values in raw corresponding to the given positions

        Args:
            positions: list of positions (x, y, z).

        Returns:
            Numpy array with the values of the voxels corresponding to each position.
            For positions outside of the atlas space `outer_value` is used if specified
            (otherwise a VoxcellError would be raised).
        '''
        voxel_idx = self.positions_to_indices(positions, outer_value is None)
        outer_mask = np.any(voxel_idx == VoxelData.OUT_OF_BOUNDS, axis=-1)
        if np.any(outer_mask):
            result = np.full(voxel_idx.shape[:-1], outer_value)
            inner_mask = np.logical_not(outer_mask)
            result[inner_mask] = self._lookup_by_indices(voxel_idx[inner_mask])
        else:
            result = self._lookup_by_indices(voxel_idx)
        return result

    def _lookup_by_indices(self, voxel_idx):
        '''values for the given voxels'''
        voxel_idx_tuple = tuple(voxel_idx.transpose())
        return self.raw[voxel_idx_tuple]

    def positions_to_indices(self, positions, strict=True):
        '''take positions, and figure out to which voxel they belong'''
        result = (positions - self.offset) / self.voxel_dimensions
        result[np.abs(result) < 1e-7] = 0.  # suppress rounding errors around 0
        result = np.floor(result).astype(np.int)
        n_dim = len(self.voxel_dimensions)
        result[result < 0] = VoxelData.OUT_OF_BOUNDS
        result[result >= self.raw.shape[:n_dim]] = VoxelData.OUT_OF_BOUNDS
        if strict and np.any(result == VoxelData.OUT_OF_BOUNDS):
            raise VoxcellError("Out of bounds position")
        return result

    def indices_to_positions(self, indices):
        ''' Return positions within given voxels

            Use fractional indices to obtain positions within voxels
            (for example, index (0.5, 0.5) would give the center of voxel (0, 0)).
        '''
        return indices * self.voxel_dimensions + self.offset

    def count(self, values):
        ''' Number of voxels with value from the given list.

            `values` could be a single value or an iterable.
        '''
        if isinstance(values, set):
            # numpy.in1d expects an array-like object as second parameter
            values = list(values)
        return np.count_nonzero(np.in1d(self.raw, values))

    def volume(self, values):
        ''' Total volume of voxels with value from the given list.

            `values` could be a single value or an iterable.
        '''
        return np.prod(self.voxel_dimensions) * self.count(values)

    def clipped(self, aabb):
        '''return a copy of this data after clipping it to an axis-aligned bounding box'''
        raw = math_utils.clip(self.raw, aabb)
        offset = aabb[0] * self.voxel_dimensions
        return VoxelData(raw, self.voxel_dimensions, self.offset + offset)

    def with_data(self, raw):
        '''return VoxelData of the same shape with different data'''
        return VoxelData(raw, self.voxel_dimensions, self.offset)


class Hierarchy(object):
    '''encapsulates data about brain structures organized in a
    hierarchical part-of relationship.'''

    def __init__(self, data):
        self.children = [Hierarchy(c) for c in data.get('children', [])]
        self.data = dict((k, data[k]) for k in set(data.keys()) - set(['children']))

    @classmethod
    def load(cls, filename):
        '''load a hierarchy of annotations in json from the Allen Brain Institute'''
        with open(filename, 'r') as f:
            return Hierarchy(json.load(f)['msg'][0])

    def find(self, attribute, value):
        '''get a list with all the subsections of a hierarchy that exactly match
        the given value on the given attribute'''
        me = [self] if self.data[attribute] == value else []
        children = (c.find(attribute, value) for c in self.children)
        return list(itertools.chain(me, *children))

    def get(self, attribute):
        '''get the set of all values of the given attribute for every subsection of a hierarchy'''
        me = [self.data[attribute]]
        children = (c.get(attribute) for c in self.children)
        return set(itertools.chain(me, *children))

    def collect(self, find_attribute, value, get_attribute):
        '''get the set of all values for the get_attribute for all sections in a hierarchy
        matching the find_attribute-value'''
        collected = (r.get(get_attribute) for r in self.find(find_attribute, value))
        return set(itertools.chain.from_iterable(collected))

    def __str__(self):
        txt = self.data.get('name', '<unnamed section>')

        if self.data:
            txt += '\n    ' + '\n    '.join('%s: %s' % (k, v)
                                            for k, v in iteritems(self.data) if k != 'name')

        if self.children:
            children = '\n'.join(str(c) for c in self.children)
            txt += '\n    children: [\n    %s\n    ]' % children.replace('\n', '\n    ')
        else:
            txt += '\n    children: []'

        return txt


class RegionMap(object):
    """ Encapsulates brain region hierarchy retrieved from NIP. """
    def __init__(self):
        self._data = dict()
        self._children = dict()
        self._parent = dict()

    def get(self, _id):
        """ Get the payload data associated with region ID. """
        return self._data[_id]

    @classmethod
    def from_dict(cls, d):
        """ Construct RegionMap from a hierarchical dictionary. """
        def include(data, parent_id):
            # pylint: disable=protected-access,missing-docstring
            _id = data.pop('id')
            if _id in result._data:
                raise VoxcellError("Duplicate id: %d" % _id)
            children = data.pop('children', [])
            result._data[_id] = data
            result._parent[_id] = parent_id
            result._children[_id] = [c['id'] for c in children]
            for c in children:
                include(c, _id)
        result = cls()
        include(copy.deepcopy(d), None)
        return result

    @classmethod
    def from_json(cls, filepath):
        """ Construct RegionMap from JSON file. """
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

    def ids(self, value, attr='name', with_descendants=True):
        """ Get set of regions matching the given attribute. """
        result = set()
        for _id, data in iteritems(self._data):
            if data.get(attr) == value:
                if with_descendants:
                    result.update(self.descendants(_id))
                else:
                    result.add(_id)
        return result

    def ascendants(self, _id):
        """ Get a list of ascendants for the given id (itself included).

            The list is sorted in level descending order.
        """
        x = _id
        result = []
        while x is not None:
            result.append(x)
            x = self._parent[x]
        return result

    def descendants(self, _id):
        """ Get a set of descendants for the given id (itself included). """
        result = set([_id])
        for c in self._children[_id]:
            result.update(self.descendants(c))
        return result

    def sub(self, _id):
        """ Return a RegionMap of region for the given id. """
        ids = self.descendants(_id)
        result = self.__class__()
        # pylint: disable=protected-access
        result._data = _sub_dict(self._data, ids)
        result._children = _sub_dict(self._children, ids)
        result._parent = _sub_dict(self._parent, ids)
        return result


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