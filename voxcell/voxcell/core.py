'''core container classes'''

import itertools
import json
import os
from collections import OrderedDict
from os.path import join as joinp
import logging

import h5py
import numpy as np
import pandas as pd
import nrrd
from voxcell import math

L = logging.getLogger(__name__)


class VoxelData(object):
    '''wrap volumetric data and some basic metadata'''

    def __init__(self, raw, voxel_dimensions, offset=None):
        '''
        Note that he units for the metadata will depend on the atlas being used.

        Args:
            raw(numpy.ndarray): actual voxel values
            voxel_dimensions(tuple of numbers): size of each voxel in space.
            offset(tuple of numbers): offset from an external atlas origin
        '''
        self.offset = offset if offset is not None else np.zeros(len(raw.shape))
        self.voxel_dimensions = voxel_dimensions
        self.raw = raw

    @classmethod
    def load_metaio(cls, mhd_path, raw_path=None):
        '''create a VoxelData object from a MetaIO file

        Args:
            mhd_path(string): path to mhd file
            raw_path(string): path to raw file, if None, .mhd file is read
                and ElementDataFile is used instead

        Returns:
            VoxelData object
        '''
        if not mhd_path.endswith('.mhd'):
            L.warning('mhd_path does not end in .mhd')

        mhd = read_mhd(mhd_path)

        if raw_path is None:
            dirname = os.path.dirname(mhd_path)
            raw_path = joinp(dirname, mhd['ElementDataFile'])

        if not raw_path.endswith('.raw'):
            L.warning('data_path does not end in .raw')

        raw = load_raw(mhd['ElementType'], mhd['DimSize'], raw_path)
        return cls(raw, mhd['ElementSpacing'])

    @classmethod
    def load_nrrd(cls, nrrd_path):
        ''' read volumetric data from a nrrd file '''
        raw, option = nrrd.read(nrrd_path)
        raw = np.reshape(raw, raw.shape, order="F")
        if 'spacing' in option:
            spacing = option['spacing']
        else:
            #TODO assert when new data will be delivered by NeuroInf.
            L.warning("spacing not defined in nrrd")
            spacing = [25, 25, 25]
        return cls(raw, spacing)

    def save_metaio(self, mhd_path, raw_filename=None):
        '''save a VoxelData header file and its accompanying data file

        Args:
            mhd_path(string): full path to mhd file
            raw_filename(string): name of raw file relative to mhd file.
                If not provided, defaults to the same as mhd_path but with the extension
                of the file extension changed to .raw
        '''
        if raw_filename is None:
            raw_filename = os.path.basename(os.path.splitext(mhd_path)[0] + '.raw')

        mhd = get_mhd_info(self.raw, raw_filename, self.voxel_dimensions, self.offset)
        save_mhd(mhd_path, mhd)
        self.raw.transpose().tofile(joinp(os.path.dirname(mhd_path), mhd['ElementDataFile']))

    def lookup(self, positions):
        '''find the values in raw corresponding to the given positions

        Args:
            positions: list of positions (x, y, z). Expected in atlas-space.

        Returns:
            Numpy array with the values of the voxels corresponding to each position
        '''
        positions = positions - self.offset
        voxel_idx = self.positions_to_indices(positions)
        voxel_idx_tuple = tuple(voxel_idx.transpose())
        return self.raw[voxel_idx_tuple]

    def positions_to_indices(self, positions):
        '''take positions, and figure out to which voxel they belong'''
        return np.floor(positions / self.voxel_dimensions).astype(np.int)

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
            filename(str): fullpath to filename to write

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
                        unique_values = np.array(f['library'][name])
                        cells.properties[name] = unique_values[data]

                    else:
                        cells.properties[name] = data

        return cells


def read_mhd(path):
    '''read a VoxelData header file'''
    with open(path) as mhd:
        data = OrderedDict((k.strip(), v.strip())
                           for k, v in (line.split('=')
                                        for line in mhd.readlines() if line.strip()))

    numerical_keys = ('CenterOfRotation', 'DimSize', 'NDims', 'ElementSpacing',
                      'Offset', 'TransformMatrix')

    for k in numerical_keys:
        if k in data and data[k] != '???':
            data[k] = tuple(int(v) for v in data[k].split())
            data[k] = data[k][0] if len(data[k]) == 1 else np.array(data[k])

    for k in data.keys():
        if isinstance(data[k], basestring):
            if data[k].lower() == 'true':
                data[k] = True
            elif data[k].lower() == 'false':
                data[k] = False

    return data


def save_mhd(path, data):
    '''save a VoxelData header file'''
    with open(path, 'w') as mhd:
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                v = ' '.join(str(x) for x in v)
            elif isinstance(v, np.ndarray):
                v = ' '.join(str(x) for x in v.flat)
            mhd.write('%s = %s\n' % (k, v))


METAIO_TO_DTYPE = {
    'MET_UCHAR': np.uint8,
    'MET_UINT': np.uint32,
    'MET_FLOAT': np.float32,
}
DTYPE_TO_METAIO = dict((v, k) for k, v in METAIO_TO_DTYPE.items())


def get_mhd_info(raw, element_datafile, voxel_dimensions, offset):
    '''Build a MetaIO header dictionary with all the elements needed for an .MHD file

    Args:
        raw(numpy.ndarray): data from which to extract sizes and types
        voxel_dimensions(numpy.ndarray): spacing of the elements
        offset(numpy.ndarray): offset from the atlas origin
        element_datafile(str): name of the corresponding datafile
        offset(tuple): (x, y, z)
    '''
    ndims = len(raw.shape)
    return {
        'ObjectType': 'Image',
        'NDims': ndims,
        'BinaryData': True,
        'BinaryDataByteOrderMSB': False,
        'TransformMatrix': np.identity(ndims),
        'Offset': np.array(offset),
        'CenterOfRotation': np.zeros(ndims),
        'AnatomicalOrientation': '???',
        'DimSize': np.array(raw.shape),
        'ElementType': DTYPE_TO_METAIO[raw.dtype.type],
        'ElementSpacing': np.array(voxel_dimensions),
        'ElementDataFile': element_datafile,
    }


def load_raw(element_type, shape, data_path):
    '''load a meta io image

    Args:
        element_type(str): a meta io element type (MET_UCHAR, MET_FLOAT)
        shape(tuple): the dimensions of the array
        data_path(str): path to the raw data file

    Returns:
        numpy array of correct type
    '''
    dtype = METAIO_TO_DTYPE[element_type]
    data = np.fromfile(data_path, dtype=dtype)

    # matlab loading code from http://help.brain-map.org/display/mousebrain/API
    # uses 'l' or 'ieee-le' which means little-endian ordering
    # uses matlab's reshape which accesses in column-major order (Fortran style)
    data = np.reshape(data, shape, order="F")
    return data
