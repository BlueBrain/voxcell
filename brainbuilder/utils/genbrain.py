'''common functionality for all circuit building tasks'''

import itertools
import json
import logging
from collections import OrderedDict
from os.path import join as joinp

import h5py
import numpy as np

L = logging.getLogger(__name__)


class MetaIO(object):
    '''wrap MetaIO files'''
    def __init__(self, mhd, raw):
        self.mhd = mhd
        self.raw = raw

    @classmethod
    def load(cls, mhd_path, raw_path):
        '''create a MetaIO object

        Args:
            mhd_path(string): path to mhd file
            raw_path(string): path to raw file

        Return:
            MetaIO object
        '''
        mhd, raw = load_meta_io(mhd_path, raw_path)
        return cls(mhd, raw)


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
            'AnatomicalOrientation': '??',
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


def load_meta_io(mhd_path, data_path):
    '''load a meta io image

    Args:
        mhd_path(str): path to .mdh file describing data
        data_path(str): path to data file described by .mhd file (usually .raw)

    Return:
        tuple of the meta information, and numpy array of correct type
    '''
    if not mhd_path.endswith('mhd'):
        L.warning('mhd_path does not end in mhd')

    if not data_path.endswith('raw'):
        L.warning('data_path does not end in raw')

    mhd = read_mhd(mhd_path)
    raw = load_raw(mhd['ElementType'], mhd['DimSize'], data_path)
    return mhd, raw


def save_meta_io(mhd_filename, mhd, raw_filename, raw):
    '''save a MetaIO header file and its accompanying data file'''
    save_mhd(mhd_filename, mhd)
    raw.transpose().tofile(raw_filename)


def load_positions(filename):
    '''load the cell positions from hdf5'''
    with h5py.File(filename, 'r') as h5:
        x = h5['x']
        y = h5['y']
        z = h5['z']
        positions = np.empty((len(x), 3))
        positions[:, 0] = x
        positions[:, 1] = y
        positions[:, 2] = z
        return positions


def save_positions(filename, positions):
    '''save the cell positions in hdf5'''
    with h5py.File(filename, 'w') as h5:
        h5.create_dataset(name='x', data=positions[:, 0])
        h5.create_dataset(name='y', data=positions[:, 1])
        h5.create_dataset(name='z', data=positions[:, 2])


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
    _, data = load_meta_io(joinp(experiment_path, experiment_type + '.mhd'),
                           joinp(experiment_path, experiment_type + '.raw'))
    return data


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


def save_orientations(filename, orientations):
    '''serialise a list of tensor fields to h5.
    orientations must be a list of 3 teson fields corresponding to the
    right vectors, up vectors and fwd vectors'''
    with h5py.File(filename, 'w') as h5:
        for name, data in zip(('right', 'up', 'fwd'), orientations):
            h5.create_dataset(name=name, data=data)


def load_orientations(filename):
    '''deserialise a list of tensor fields from h5'''
    with h5py.File(filename, 'r') as h5:
        return [np.array(h5[name]) for name in ('right', 'up', 'fwd')]


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


def gcd(a, b):
    '''Return greatest common divisor using Euclid's Algorithm.'''
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    '''Return lowest common multiple.'''
    return a * b // gcd(a, b)


# TODO consider making this a np ufunc
def lcmm(args):
    '''Return lcm of args.'''
    return reduce(lcm, args)
