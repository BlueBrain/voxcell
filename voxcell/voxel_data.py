""" Access to volumetric data. """

import numpy as np
import nrrd

from voxcell import math_utils, deprecate
from voxcell.exceptions import VoxcellError
from voxcell.quaternion import quaternions_to_matrices


def _pivot_axes(a, k):
    """
    Move `k` first dimensions of `a` to the end, preserving their order.

    I.e., _pivot_axes(A x B x C x D x E, 2) -> C x D x E x A x B
    """
    n = len(a.shape)
    assert 0 <= k <= n
    return np.moveaxis(a, np.arange(k), np.arange(n - k, n))


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

        if offset is None:
            self.offset = np.zeros(self.ndim)
        else:
            offset = np.array(offset, dtype=np.float32)
            if offset.shape != (self.ndim,):
                raise VoxcellError("'offset' shape should be: {0} (got: {1})".format(
                    (self.ndim,), offset.shape
                ))
            self.offset = offset

        if len(raw.shape) < self.ndim:
            raise VoxcellError("'raw' should have at least {0} dimensions (got: {1})".format(
                self.ndim, len(raw.shape)
            ))
        self.raw = raw

    @property
    def voxel_volume(self):
        """ Voxel volume. """
        return abs(np.product(self.voxel_dimensions))

    @property
    def ndim(self):
        """ Number of dimensions. """
        return len(self.voxel_dimensions)

    @property
    def shape(self):
        """ Number of voxels in each dimension. """
        return self.raw.shape[:self.ndim]

    @property
    def payload_shape(self):
        """ Shape of the data stored per voxel. """
        return self.raw.shape[self.ndim:]

    @property
    def bbox(self):
        """ Bounding box. """
        return np.array([self.offset,
                         self.offset + self.voxel_dimensions * self.shape])

    @classmethod
    def load_nrrd(cls, nrrd_path):
        ''' read volumetric data from a nrrd file '''
        data, options = nrrd.read(nrrd_path)

        # According to http://teem.sourceforge.net/nrrd/format.html#spacedirections,
        # 'space directions' could use 'none' for "payload" axes.
        # As we need space directions only for "space" axes, we rely on either
        # 'space dimension' or 'space' header to slice them out.
        # NB: only a subset of possible 'space' values is supported at the moment.
        if 'space dimension' in options:
            ndim = options['space dimension']
        elif 'space' in options:
            ndim = {
                'right-anterior-superior': 3, 'RAS': 3,
                'left-anterior-superior': 3, 'LAS': 3,
                'left-posterior-superior': 3, 'LPS': 3,
            }[options['space']]
        else:
            ndim = 0  # use all 'space directions'

        if 'space directions' in options:
            directions = np.array(options['space directions'][-ndim:], dtype=np.float32)
            if not math_utils.is_diagonal(directions):
                raise NotImplementedError("Only diagonal space directions supported at the moment")
            spacings = directions.diagonal()
        elif 'spacings' in options:
            spacings = np.array(options['spacings'][-ndim:], dtype=np.float32)
        else:
            raise VoxcellError("spacings not defined in nrrd")

        offset = None
        if 'space origin' in options:
            offset = np.array(options['space origin'], dtype=np.float32)

        # In NRRD 'payload' axes go first, move them to the end
        raw = _pivot_axes(data, len(data.shape) - len(spacings))

        return cls(raw, spacings, offset)

    def save_nrrd(self, nrrd_path):
        '''save a VoxelData to an nrrd file

        Args:
            nrrd_path(string): full path to nrrd file
        '''
        # from http://teem.sourceforge.net/nrrd/format.html#space
        options = {
            'space dimension': self.ndim,
            'space directions': np.diag(self.voxel_dimensions),
            'space origin': self.offset,
        }
        # In NRRD 'payload' axes should go first, move them to the beginning
        nrrd_data = _pivot_axes(self.raw, self.ndim)
        nrrd.write(nrrd_path, nrrd_data, options=options)

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
        result[result < 0] = VoxelData.OUT_OF_BOUNDS
        result[result >= self.shape] = VoxelData.OUT_OF_BOUNDS
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
        return self.count(values) * self.voxel_volume

    def clipped(self, aabb):
        '''return a copy of this data after clipping it to an axis-aligned bounding box'''
        deprecate.fail("Deprecated. Please use VoxelData.clip() instead.")
        raw = math_utils.clip(self.raw, aabb)
        offset = aabb[0] * self.voxel_dimensions
        return VoxelData(raw, self.voxel_dimensions, self.offset + offset)

    def clip(self, bbox, na_value=0, inplace=False):
        """ Assign `na_value` to voxels outside of axis-aligned bounding box.

            Args:
                bbox: bounding box in real-world coordinates
                na_value: value to use for voxels outside of bbox
                inplace(bool): modify data inplace

            Returns:
                None if `inplace` is True, new VoxelData otherwise
        """
        bbox = np.array(bbox)
        if bbox.shape != (2, self.ndim):
            raise VoxcellError("Invalid bbox shape: {}".format(bbox.shape))

        aabb = ((bbox - self.offset) / self.voxel_dimensions).astype(np.int)

        # ensure clipped volume is inside bbox
        aa, bb = np.clip(aabb, np.full(self.ndim, -1), self.shape)
        aa += 1
        bb -= 1
        if np.any(aa > bb):
            raise VoxcellError("Empty slice")

        indices = [range(a, b + 1) for a, b in zip(aa, bb)]

        if inplace:
            mask = np.full_like(self.raw, False, dtype=bool)
            mask[indices] = True
            self.raw[np.logical_not(mask)] = na_value
        else:
            raw = np.full_like(self.raw, na_value)
            raw[indices] = self.raw[indices]
            return VoxelData(raw, self.voxel_dimensions, self.offset)

    def filter(self, predicate, inplace=False):
        """ Set values for voxel positions not satisfying `predicate` to zero.

            Args:
                predicate: (x_1,..,x_k) -> bool
                inplace(bool): modify data inplace

            Returns:
                None if `inplace` is True, new VoxelData otherwise
        """
        ijk = np.stack(np.mgrid[[slice(0, d) for d in self.shape]], axis=-1)
        xyz = self.indices_to_positions(0.5 + ijk)
        mask = np.apply_along_axis(predicate, -1, xyz)
        if inplace:
            self.raw[np.invert(mask)] = 0
        else:
            raw = np.zeros_like(self.raw)
            raw[mask] = self.raw[mask]
            return VoxelData(raw, self.voxel_dimensions, self.offset)

    def compact(self, na_values=(0,), inplace=False):
        """
        Reduce size of raw data by clipping N/A values.

        Args:
            na_values(tuple): values to clip
            inplace(bool): modify data inplace

        Returns:
            None if `inplace` is True, new VoxelData otherwise
        """
        mask = np.logical_not(np.isin(self.raw, na_values))
        aabb = math_utils.minimum_aabb(mask)

        raw = math_utils.clip(self.raw, aabb)
        offset = self.indices_to_positions(aabb[0])

        if inplace:
            self.raw = raw
            self.offset = offset
        else:
            return VoxelData(raw, self.voxel_dimensions, offset)

    def with_data(self, raw):
        '''return VoxelData of the same shape with different data'''
        return VoxelData(raw, self.voxel_dimensions, self.offset)


class OrientationField(VoxelData):
    """
    Volumetric data with rotation per voxel.

    See also:
        https://bbpteam.epfl.ch/project/spaces/display/NRINF/Orientation+Field
    """
    def __init__(self, *args, **kwargs):
        super(OrientationField, self).__init__(*args, **kwargs)
        if self.raw.dtype not in (np.int8, np.float32, np.float64):
            raise VoxcellError("Invalid volumetric data dtype: %s" % self.raw.dtype)
        if self.payload_shape != (4,):
            raise VoxcellError("Volumetric data should store (x, y, z, w) tuple per voxel")

    # pylint: disable=arguments-differ
    def lookup(self, positions):
        """
        Orientations corresponding to the given positions.

        Args:
            positions: list of positions (x, y, z).

        Returns:
            Numpy array with the rotation matrices corresponding to each position.
        """
        result = super(OrientationField, self).lookup(positions, outer_value=None)

        # normalize int8 data
        if result.dtype == np.int8:
            result = result / 127.0

        # change quaternion component order: (w, x, y, z) -> (x, y, z, w)
        result = np.roll(result, -1, axis=-1)

        return quaternions_to_matrices(result)
