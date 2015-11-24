'''common functionality for all circuit building tasks'''

import itertools
import json
import logging
import os

from collections import OrderedDict
from os.path import join as joinp

import h5py
import numpy as np
import pandas as pd

from brainbuilder.utils import math

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

        Return:
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

    def save_metaio(self, mhd_path, raw_filename=None):
        '''save a VoxelData header file and its accompanying data file

        Args:
            mhd_path(string): full path to mhd file
            raw_filename(string): name of raw file relative to mhd file.
                If not provided, defaults to the same as mhd_path but with the extension
                of the file changed to .raw
        '''
        if raw_filename is None:
            raw_filename = os.path.splitext(mhd_path)[0] + '.raw'

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
        voxel_idx = cell_positions_to_voxel_indices(positions, self.voxel_dimensions)
        voxel_idx_tuple = tuple(voxel_idx.transpose())
        return self.raw[voxel_idx_tuple]


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

# conversion of types in .mhd file to numpy types
METAIO_TO_DTYPE = {
    'MET_UCHAR': np.uint8,
    'MET_UINT': np.uint32,
    'MET_FLOAT': np.float32,
}

# conversion of types numpy types to type in .mhd file
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

    Return:
        numpy array of correct type
    '''
    dtype = METAIO_TO_DTYPE[element_type]
    data = np.fromfile(data_path, dtype=dtype)

    # matlab loading code from http://help.brain-map.org/display/mousebrain/API
    # uses 'l' or 'ieee-le' which means little-endian ordering
    # uses matlab's reshape which accesses in column-major order (Fortran style)
    data = np.reshape(data, shape, order="F")
    return data


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


def load_trace_data(experiment_path, experiment_type):
    '''load the experiment pointed to by experiment_path, of experiment_type
        ('injection', 'energy', 'density', 'intensity', )
        the data is clipped, and returned
    '''
    metaio = VoxelData.load_metaio(joinp(experiment_path, experiment_type + '.mhd'),
                                   joinp(experiment_path, experiment_type + '.raw'))
    return metaio.raw


def create_voxel_cube(max_x, max_y, max_z, dtype=object):
    '''create voxel cube used to store the experiments that contain data for a
       particular voxel value

    Return:
        Empty array of [max_x][max_y][max_z]
    '''
    return np.empty((max_x, max_y, max_z), dtype=dtype)


def gen_indices(numpy_array):
    '''create indices for the numpy_array

    Args:
        numpy_array(np.array): The array over which the indices are created

    Return:
        An iterator that goes over the full mesh.  The iterator returns triples of (x, y, z)

    '''
    (max_x, max_y, max_z) = numpy_array.shape
    return itertools.product(xrange(max_x), xrange(max_y), xrange(max_z))


def get_popular_voxels(f):
    '''returns a list of (# of, (x, y, z)) if # > 0'''
    return reversed(sorted((len(f[x][y][z]), (x, y, z))
                           for x, y, z in gen_indices(f) if f[x][y][z]))


def get_regions_mask_by_ids(annotation_raw, region_ids):
    '''get a binary voxel mask where the voxel belonging to the given region ids are True'''

    in_region = np.in1d(annotation_raw, list(region_ids))
    in_region = in_region.reshape(np.shape(annotation_raw))
    return in_region


def get_regions_mask_by_names(annotation_raw, hierarchy, names):
    '''get a binary voxel mask where the voxel belonging to the given region names are True'''
    all_ids = []
    for n in names:
        ids = hierarchy.collect('name', n, 'id')
        if not ids:
            raise KeyError(n)
        all_ids.extend(ids)

    return get_regions_mask_by_ids(annotation_raw, all_ids)


def get_points_list_from_mask(mask):
    '''get all the voxel indices for positive values of the binary voxel mask
    for example: for a mask that has everything to true, the result will look like:
    [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 1],
           ...]
    '''
    return np.array(np.nonzero(mask)).swapaxes(0, 1)


def cell_voxel_indices_to_positions(cell_voxel_indices, voxel_dimensions):
    '''create random position in the selected voxel. Add some random jitter'''
    jitter = np.random.random(np.shape(cell_voxel_indices))
    return ((cell_voxel_indices + 1) - jitter) * np.array(voxel_dimensions)


def cell_positions_to_voxel_indices(positions, voxel_dimensions):
    '''take positions, and figure out to which voxel they belong'''
    return np.floor(positions / voxel_dimensions).astype(np.int)


def build_cell_density_from_positions(positions, voxel_dimensions, dtype=np.uint8):
    '''calculate cell density from the cell positions'''
    if positions.shape[0] == 0:
        return VoxelData(np.zeros([1] * len(voxel_dimensions), dtype=dtype), voxel_dimensions)

    else:
        aabb_min, aabb_max = get_positions_minimum_aabb(positions)

        dimensions = np.floor((aabb_max - aabb_min) / voxel_dimensions).astype(np.uint)
        dimensions += np.ones_like(dimensions)

        positions = positions - aabb_min
        voxel_indices = cell_positions_to_voxel_indices(positions, voxel_dimensions)

        density = np.zeros(dimensions, dtype=dtype)
        for x, y, z in voxel_indices:
            density[x, y, z] += 1

        return VoxelData(density, voxel_dimensions, offset=aabb_min)


def build_homogeneous_density(mask, voxel_dimensions, offset=None, value=255):
    '''build an artificial homogeneous density'''
    raw = np.zeros(mask.shape, dtype=np.uint8)
    raw[mask] = value
    return VoxelData(raw, voxel_dimensions=voxel_dimensions, offset=offset)


def build_layered_annotation(shape, heights, layer_ids):
    ''' build an artificial annotation composed of layers along the Y axis
    Args:
        shape: 2-tuple with the size of the resulting array in X and Z in number of voxels
        heights: sequence of layer heights in number of voxels from lower to higher layer
        layer_ids: sequence of layer ids ordered from lower to higher layer
    '''
    assert len(layer_ids) == len(heights)
    boundaries = np.zeros(len(heights) + 1, dtype=np.uint)
    boundaries[1:] = np.cumsum(heights)

    raw = np.zeros((shape[0], boundaries[-1], shape[1]), dtype=np.uint32)

    idx = 0
    for i, j in zip(boundaries[:-1], boundaries[1:]):
        raw[:, i:j, :] = layer_ids[idx]
        idx += 1

    return raw


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
                    f.create_dataset('library/' + name, data=unique_values)

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


def get_minimum_aabb(mask):
    '''calculate the minimum axis-aligned bounding box for a volume mask
    Returns:
        A tuple containing the minimum x,y,z and maximum x,y,z
    '''
    idx = np.nonzero(mask)
    return np.min(idx, axis=1), np.max(idx, axis=1)


def get_positions_minimum_aabb(positions):
    '''calculate the minimum axis-aligned bounding box for a list of positions
    Returns:
        A tuple containing the minimum x,y,z and maximum x,y,z
    '''
    return np.min(positions, axis=0), np.max(positions, axis=0)


def clip(mask, aabb):
    '''take a numpy array it to an axis-aligned bounding box'''
    idx = [slice(s, e + 1) for s, e in zip(*aabb)]
    return mask[idx]


def clip_volume(density, aabb):
    '''take a density and clip it to an axis-aligned bounding box'''
    raw = clip(density.raw, aabb)
    offset = aabb[0] * density.voxel_dimensions
    return VoxelData(raw, density.voxel_dimensions, density.offset + offset)
