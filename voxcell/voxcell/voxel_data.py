""" Access to volumetric data. """

import numpy as np
import nrrd

from voxcell import math_utils
from voxcell.exceptions import VoxcellError


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
