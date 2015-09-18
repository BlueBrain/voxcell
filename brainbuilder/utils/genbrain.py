'''common functionality for all circuit building tasks'''

import itertools
import json
import logging
import os
import copy

from collections import OrderedDict
from os.path import join as joinp

import h5py
import numpy as np
import pandas as pd

from brainbuilder.utils import math

L = logging.getLogger(__name__)


class MetaIO(object):
    '''wrap MetaIO files'''
    def __init__(self, mhd, raw):
        self.mhd = mhd
        self.raw = raw

    @classmethod
    def load(cls, mhd_path, raw_path=None):
        '''create a MetaIO object

        Args:
            mhd_path(string): path to mhd file
            raw_path(string): path to raw file, if None, .mhd file is read
                and ElementDataFile is used instead

        Return:
            MetaIO object
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
        return cls(mhd, raw)

    def save(self, mhd_path, raw_filename=None):
        '''save a MetaIO header file and its accompanying data file

        Args:
            mhd_path(string): full path to mhd file
            raw_filename(string): name of raw file relative to mhd file.
                Optional: if None, ElementDataFile in mhd is used instead
        '''
        if raw_filename is not None:
            mhd = copy.copy(self.mhd)
            mhd['ElementDataFile'] = raw_filename
        else:
            mhd = self.mhd

        save_mhd(mhd_path, mhd)
        self.raw.transpose().tofile(joinp(os.path.dirname(mhd_path), mhd['ElementDataFile']))


def read_mhd(path):
    '''read a MetaIO header file'''
    with open(path) as mhd:
        data = OrderedDict((k.strip(), v.strip())
                           for k, v in (line.split('=')
                                        for line in mhd.readlines() if line.strip()))

    numerical_keys = ('CenterOfRotation', 'DimSize', 'NDims', 'ElementSpacing',
                      'Offset', 'TransformMatrix')

    for k in numerical_keys:
        if k in data and data[k] != '???':
            data[k] = tuple(int(v) for v in data[k].split())
            data[k] = data[k][0] if len(data[k]) == 1 else data[k]

    for k in data.keys():
        if isinstance(data[k], basestring):
            if data[k].lower() == 'true':
                data[k] = True
            elif data[k].lower() == 'false':
                data[k] = False

    return data


def save_mhd(path, data):
    '''save a MetaIO header file'''
    with open(path, 'w') as mhd:
        for k, v in data.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                v = ' '.join(str(x) for x in v)
            mhd.write('%s = %s\n' % (k, v))

# conversion of types in .mhd file to numpy types
METAIO_TO_DTYPE = {'MET_UCHAR': np.uint8,
                   'MET_UINT': np.uint32,
                   'MET_FLOAT': np.float32,
                   }
# conversion of types numpy types to type in .mhd file
DTYPE_TO_MATAIO = dict((v, k) for k, v in METAIO_TO_DTYPE.items())


def get_mhd_info(dimensions, element_type, element_spacing, element_datafile):
    '''Get a dictionary with all the elements needed for an .MHD file

    Args:
        dimensions(tuple): (x, y, z)
        element_type(numpy type): Type of the data
        element_spacing(tuple): spacing of the elements
        element_datafile(str): name of the corresponding datafile
    '''
    return {'ObjectType': 'Image',
            'NDims': 3,
            'BinaryData': True,
            'BinaryDataByteOrderMSB': False,
            'CompressedData': False,
            'TransformMatrix': '1 0 0 0 1 0 0 0 1',
            'Offset': '0 0 0',
            'CenterOfRotation': '0 0 0',
            'AnatomicalOrientation': '???',
            'DimSize': ' '.join(str(d) for d in dimensions),
            'ElementType': DTYPE_TO_MATAIO[element_type],
            'ElementSpacing': ' '.join(str(e) for e in element_spacing),
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


def load_hierarchy(filename):
    '''load a hierarchy of annotations in json from the Allen Brain Institute'''
    return json.load(file(filename))


def find_in_hierarchy(hierarchy, attribute, value):
    '''get a list with all the subsections of a hierarchy that exactly match
    the given value on the given attribute'''
    if hierarchy[attribute] == value:
        return [hierarchy]
    else:
        found = [find_in_hierarchy(c, attribute, value) for c in hierarchy['children']]
        res = []
        for c in found:
            if c:
                res.extend(c)
        return res


def get_in_hierarchy(hierarchy, attribute):
    '''get the list of all values of the given attribute for every subsection of a hierarchy'''
    res = [hierarchy[attribute]]
    for c in hierarchy['children']:
        res.extend(get_in_hierarchy(c, attribute))
    return res


def collect_in_hierarchy(hierarchy, find_attribute, value, get_attribute):
    '''get the list of all values for the get_attribute for all sections ina hierarchy
    matching the find_attribute-value'''
    all_got = []

    found = find_in_hierarchy(hierarchy, find_attribute, value)
    for r in found:
        got = get_in_hierarchy(r, get_attribute)
        all_got.extend(got)

    return all_got


def load_trace_data(experiment_path, experiment_type):
    '''load the experiment pointed to by experiment_path, of experiment_type
        ('injection', 'energy', 'density', 'intensity', )
        the data is clipped, and returned
    '''
    metaio = MetaIO.load(joinp(experiment_path, experiment_type + '.mhd'),
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

    in_region = np.in1d(annotation_raw, region_ids)
    in_region = in_region.reshape(np.shape(annotation_raw))
    return in_region


def get_regions_mask_by_names(annotation_raw, hierarchy, names):
    '''get a binary voxel mask where the voxel belonging to the given region names are True'''
    all_ids = []
    for n in names:
        ids = collect_in_hierarchy(hierarchy, 'name', n, 'id')
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


def cell_density_from_positions(positions, density_dimensions, voxel_dimensions, dtype=np.uint8):
    '''calculate cell density from the cell positions'''
    voxel_indices = cell_positions_to_voxel_indices(positions, voxel_dimensions)

    density = np.zeros(density_dimensions, dtype=dtype)
    for coords in voxel_indices:
        density[coords[0], coords[1], coords[2]] += 1

    return density


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
                    f.create_dataset('cells/properties/' + name, data=indices)
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
