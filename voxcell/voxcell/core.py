'''core container classes'''

import itertools
import json
import logging

import h5py
import numpy as np
import pandas as pd
import nrrd

from voxcell import math, VoxcellError

L = logging.getLogger(__name__)


def require_shape(a, shape):
    """ Raise VoxcellError if `a.shape` mismatches `shape`. """
    if a.shape != shape:
        raise VoxcellError("Shape mismatch: expected {0}, actual {1}".format(shape, a.shape))


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
        n_dim = len(raw.shape)

        if offset is None:
            offset = np.zeros(n_dim)
        else:
            offset = np.array(offset, dtype=np.float32)

        voxel_dimensions = np.array(voxel_dimensions, dtype=np.float32)

        require_shape(offset, (n_dim,))
        require_shape(voxel_dimensions, (n_dim,))

        self.offset = offset
        self.voxel_dimensions = voxel_dimensions
        self.raw = raw

    @classmethod
    def load_nrrd(cls, nrrd_path):
        ''' read volumetric data from a nrrd file '''
        raw, option = nrrd.read(nrrd_path)

        if 'space directions' in option:
            directions = np.array(option['space directions'], dtype=np.float32)
            if not math.is_diagonal(directions):
                raise NotImplementedError("Only diagonal space directions supported at the moment")
            spacings = 1.0 / directions.diagonal()
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
        options = {'spacings': self.voxel_dimensions,
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
        result = np.round((positions - self.offset) / self.voxel_dimensions, 3)
        result = np.floor(result).astype(np.int)
        result[result < 0] = VoxelData.OUT_OF_BOUNDS
        result[result >= self.raw.shape] = VoxelData.OUT_OF_BOUNDS
        if strict and np.any(result == VoxelData.OUT_OF_BOUNDS):
            raise VoxcellError("Out of bounds position")
        return result

    def clipped(self, aabb):
        '''return a copy of this data after clipping it to an axis-aligned bounding box'''
        raw = math.clip(self.raw, aabb)
        offset = aabb[0] * self.voxel_dimensions
        return VoxelData(raw, self.voxel_dimensions, self.offset + offset)


class Hierarchy(object):
    '''encapsulates data about brain structures organized in a
    hierarchical part-of relationship.'''

    def __init__(self, data):
        self.children = [Hierarchy(c) for c in data.get('children', [])]
        self.data = dict((k, data[k]) for k in set(data.keys()) - set(['children']))

    @classmethod
    def load(cls, filename):
        '''load a hierarchy of annotations in json from the Allen Brain Institute'''
        return Hierarchy(json.load(file(filename))['msg'][0])

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
                                            for k, v in self.data.iteritems() if k != 'name')

        if self.children:
            children = '\n'.join(str(c) for c in self.children)
            txt += '\n    children: [\n    %s\n    ]' % children.replace('\n', '\n    ')
        else:
            txt += '\n    children: []'

        return txt


class CellCollection(object):
    '''Encapsulates all the data related to a collection of cells that compose a circuit.

    Multi-dimensional properties (such as positions and orientations) are attributes.
    General properties are a in a pandas DataFrame object "properties".
    '''
    def __init__(self):
        self.positions = None
        self.orientations = None
        self.properties = pd.DataFrame()

    def add_properties(self, new_properties):
        '''adds new columns to the properties DataFrame

        Args:
            new_properties: a pandas DataFrame object
        '''
        self.properties = pd.concat([self.properties, new_properties], axis=1)

    def remove_unassigned_cells(self):
        ''' remove cells with one or more unassigned property '''
        idx_unassigned = self.properties[self.properties.isnull().any(axis=1)].index
        self.properties = self.properties.drop(idx_unassigned)
        self.orientations = np.delete(self.orientations, idx_unassigned, 0)
        self.positions = np.delete(self.positions, idx_unassigned, 0)

    def save(self, filename):
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
                                 data=math.matrices_to_quaternions(self.orientations))

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
                    # doing a cast to np.str changes the column to be fixed-size strings
                    # (automatically set to the current maximum).
                    data = data.astype(np.str)

                    unique_values, indices = np.unique(data, return_inverse=True)
                    f.create_dataset('cells/properties/' + name, data=indices.astype(np.uint32))
                    dt = h5py.special_dtype(vlen=unicode)
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
            data = f['cells']
            if 'positions' in data:
                cells.positions = np.array(data['positions'])

            if 'orientations' in data:
                cells.orientations = np.array(data['orientations'])
                cells.orientations = math.quaternions_to_matrices(cells.orientations)

            if 'properties' in data:
                properties = data['properties']

                for name, data in properties.iteritems():
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
