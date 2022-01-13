"""Access to volumetric data."""
from functools import reduce

import nrrd
import numpy as np
from numpy.testing import assert_array_equal

from voxcell import math_utils
from voxcell.exceptions import VoxcellError
from voxcell.quaternion import quaternions_to_matrices


def _pivot_axes(a, k):
    """Move `k` first dimensions of `a` to the end, preserving their order.

    I.e., _pivot_axes(A x B x C x D x E, 2) -> C x D x E x A x B
    """
    n = len(a.shape)
    assert 0 <= k <= n
    return np.moveaxis(a, np.arange(k), np.arange(n - k, n))


class VoxelData:
    """Wrap volumetric data and some basic metadata."""

    OUT_OF_BOUNDS = -1

    def __init__(self, raw, voxel_dimensions, offset=None):
        """Note that he units for the metadata will depend on the atlas being used.

        Args:
            raw(numpy.ndarray): actual voxel values
            voxel_dimensions(tuple of numbers): size of each voxel in space.
            offset(tuple of numbers): offset from an external atlas origin
        """
        voxel_dimensions = np.array(voxel_dimensions, dtype=np.float32)
        if len(voxel_dimensions.shape) > 1:
            raise VoxcellError(
                f"voxel_dimensions should be a 1-d array (got: {len(voxel_dimensions.shape)})")
        self.voxel_dimensions = voxel_dimensions

        if offset is None:
            self.offset = np.zeros(self.ndim, dtype=np.float32)
        else:
            offset = np.array(offset, dtype=np.float32)
            if offset.shape != (self.ndim,):
                raise VoxcellError(
                    f"'offset' shape should be: {(self.ndim,)} (got: {offset.shape})")
            self.offset = offset

        if len(raw.shape) < self.ndim:
            raise VoxcellError(
                f"'raw' should have at least {self.ndim} dimensions (got: {len(raw.shape)})")
        self.raw = raw

    @property
    def voxel_volume(self):
        """Voxel volume."""
        return abs(np.product(self.voxel_dimensions))

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.voxel_dimensions)

    @property
    def shape(self):
        """Number of voxels in each dimension."""
        return self.raw.shape[:self.ndim]

    @property
    def payload_shape(self):
        """Shape of the data stored per voxel."""
        return self.raw.shape[self.ndim:]

    @property
    def bbox(self):
        """Bounding box."""
        return np.array([self.offset,
                         self.offset + self.voxel_dimensions * self.shape])

    @classmethod
    def load_nrrd(cls, nrrd_path):
        """Read volumetric data from a nrrd file.

        Args:
            nrrd_path (str): path to the nrrd file.
        """
        data, header = nrrd.read(nrrd_path)

        # According to http://teem.sourceforge.net/nrrd/format.html#spacedirections,
        # 'space directions' could use 'none' for "payload" axes.
        # As we need space directions only for "space" axes, we rely on either
        # 'space dimension' or 'space' header to slice them out.
        # NB: only a subset of possible 'space' values is supported at the moment.
        if 'space dimension' in header:
            ndim = header['space dimension']
        elif 'space' in header:
            ndim = {
                'right-anterior-superior': 3, 'RAS': 3,
                'left-anterior-superior': 3, 'LAS': 3,
                'left-posterior-superior': 3, 'LPS': 3,
                'posterior-inferior-right': 3, 'PIR': 3,
            }[header['space']]
        else:
            ndim = 0  # use all 'space directions'

        if 'space directions' in header:
            directions = np.array(header['space directions'][-ndim:], dtype=np.float32)
            if not math_utils.is_diagonal(directions):
                raise NotImplementedError("Only diagonal space directions supported at the moment")
            spacings = directions.diagonal()
        elif 'spacings' in header:
            spacings = np.array(header['spacings'][-ndim:], dtype=np.float32)
        else:
            raise VoxcellError("spacings not defined in nrrd")

        offset = None
        if 'space origin' in header:
            offset = np.array(header['space origin'], dtype=np.float32)

        # In NRRD 'payload' axes go first, move them to the end
        raw = _pivot_axes(data, len(data.shape) - len(spacings))

        return cls(raw, spacings, offset)

    def save_nrrd(self, nrrd_path, encoding=None):
        """Save a VoxelData to an nrrd file.

        Args:
            nrrd_path(string): full path to nrrd file
            encoding(string): encoding option to save as
        """
        # from http://teem.sourceforge.net/nrrd/format.html#space
        space_directions = np.diag(self.voxel_dimensions)
        dim_defect = len(self.raw.shape) - self.ndim
        if dim_defect > 0:
            # The nrrd specifications require that
            # we prepend a None for each of the extra axes
            # before specifying the volume 3D axes.
            # For instance, a volume of orientations (3D direction vectors or quaternions)
            # or of RGB colors (3D int vectors) requires an initial None value.
            space_directions = [None] * dim_defect + list(space_directions)
        header = {
            'space dimension': self.ndim,
            'space directions': space_directions,
            'space origin': self.offset,
        }

        # Case of an N-dimensional vector field over a volume
        if dim_defect == 1 and np.issubdtype(self.raw.dtype, np.number):
            # Required by ITK-SNAP (http://www.itksnap.org/pmwiki/pmwiki.php) and 3D Slicer
            # (https://download.slicer.org/) to render 3D-vector fields using a RGB field.
            # Note that ITK-SNAP raises a "Failed to load segmentation error" but lets you watch
            # the volumetric data.
            header['kinds'] = ['vector', 'domain', 'domain', 'domain']

        if encoding is not None:
            header['encoding'] = encoding

        # In NRRD 'payload' axes should go first, move them to the beginning
        nrrd_data = _pivot_axes(self.raw, self.ndim)
        nrrd.write(nrrd_path, nrrd_data, header=header)

    def lookup(self, positions, outer_value=None):
        """Find the values in raw corresponding to the given positions.

        Args:
            positions: list of positions (x, y, z).
            outer_value: value to be returned for positions outside the atlas space.
                If `None`, a VoxcellError is raised in that case.

        Returns:
            Numpy array with the values of the voxels corresponding to each position.
        """
        voxel_idx = self.positions_to_indices(positions, strict=outer_value is None)
        outer_mask = np.any(voxel_idx == VoxelData.OUT_OF_BOUNDS, axis=-1)
        if np.any(outer_mask):
            result = np.full(voxel_idx.shape[:-1], outer_value)
            inner_mask = np.logical_not(outer_mask)  # pylint: disable=assignment-from-no-return
            result[inner_mask] = self._lookup_by_indices(voxel_idx[inner_mask])
        else:
            result = self._lookup_by_indices(voxel_idx)
        return result

    def _lookup_by_indices(self, voxel_idx):
        """Values for the given voxels."""
        voxel_idx_tuple = tuple(voxel_idx.transpose())
        return self.raw[voxel_idx_tuple]

    def positions_to_indices(self, positions, strict=True, keep_fraction=False):
        """Take positions, and the index of the voxel to which they belong.

        Args:
            positions(np.array of Nx3): positions in voxel volume
            strict(bool): raise VoxcellError if any of the positions are out of bounds
            keep_fraction(bool): keep the fractional portion of the positions

        Returns:
            np.array(Nx3) with the voxels coordinates corresponding to each position.

        """
        result = (positions - self.offset) / self.voxel_dimensions
        result[np.abs(result) < 1e-7] = 0.  # suppress rounding errors around 0
        if not keep_fraction:
            result = np.floor(result).astype(int)
        result[result < 0] = VoxelData.OUT_OF_BOUNDS
        result[result >= self.shape] = VoxelData.OUT_OF_BOUNDS
        if strict and np.any(result == VoxelData.OUT_OF_BOUNDS):
            raise VoxcellError("Out of bounds position")
        return result

    def indices_to_positions(self, indices):
        """Return positions within given voxels.

        Use fractional indices to obtain positions within voxels
        (for example, index (0.5, 0.5) would give the center of voxel (0, 0)).
        """
        return indices * self.voxel_dimensions + self.offset

    def count(self, values):
        """Number of voxels with value from the given list.

        `values` could be a single value or an iterable.
        """
        if isinstance(values, set):
            # numpy.in1d expects an array-like object as second parameter
            values = list(values)
        return np.count_nonzero(np.in1d(self.raw, values))

    def volume(self, values):
        """Total volume of voxels with value from the given list.

        `values` could be a single value or an iterable.
        """
        return self.count(values) * self.voxel_volume

    def clip(self, bbox, na_value=0, inplace=False):
        """Assign `na_value` to voxels outside of axis-aligned bounding box.

        Args:
            bbox: bounding box in real-world coordinates
            na_value: value to use for voxels outside of bbox
            inplace(bool): modify data inplace

        Returns:
            None if `inplace` is True, new VoxelData otherwise
        """
        bbox = np.array(bbox)
        if bbox.shape != (2, self.ndim):
            raise VoxcellError(f"Invalid bbox shape: {bbox.shape}")

        aabb = ((bbox - self.offset) / self.voxel_dimensions).astype(int)

        # ensure clipped volume is inside bbox
        aa, bb = np.clip(aabb, np.full(self.ndim, -1), self.shape)
        aa += 1
        bb -= 1
        if np.any(aa > bb):
            raise VoxcellError("Empty slice")

        indices = tuple(range(a, b + 1) for a, b in zip(aa, bb))

        if inplace:
            mask = np.full_like(self.raw, False, dtype=bool)
            mask[indices] = True
            self.raw[np.logical_not(mask)] = na_value
            return None
        else:
            raw = np.full_like(self.raw, na_value)
            raw[indices] = self.raw[indices]
            return VoxelData(raw, self.voxel_dimensions, self.offset)

    def filter(self, predicate, inplace=False):
        """Set values for voxel positions not satisfying `predicate` to zero.

        Args:
            predicate: N x k [float] -> N x 1 [bool]
            inplace(bool): modify data inplace

        Returns:
            None if `inplace` is True, new VoxelData otherwise
        """
        ijk = np.stack(np.mgrid[[slice(0, d) for d in self.shape]], axis=-1)
        xyz = self.indices_to_positions(0.5 + ijk)
        mask = predicate(xyz.reshape(-1, self.ndim)).reshape(self.shape)
        if inplace:
            self.raw[np.invert(mask)] = 0
            return None
        else:
            raw = np.zeros_like(self.raw)
            raw[mask] = self.raw[mask]
            return VoxelData(raw, self.voxel_dimensions, self.offset)

    def compact(self, na_values=(0,), inplace=False):
        """Reduce size of raw data by clipping N/A values.

        Args:
            na_values(tuple): values to clip
            inplace(bool): modify data inplace

        Returns:
            None if `inplace` is True, new VoxelData otherwise
        """
        mask = np.logical_not(  # pylint: disable=assignment-from-no-return
            math_utils.isin(self.raw, na_values)
        )
        aabb = math_utils.minimum_aabb(mask)

        raw = math_utils.clip(self.raw, aabb)
        offset = self.indices_to_positions(aabb[0])

        if inplace:
            self.raw = raw
            self.offset = offset
            return None
        else:
            return VoxelData(raw, self.voxel_dimensions, offset)

    def with_data(self, raw):
        """Return VoxelData of the same shape with different data."""
        return VoxelData(raw, self.voxel_dimensions, self.offset)

    @staticmethod
    def reduce(function, iterable):
        """Return a VoxelData by reducing the raw contents of the VoxelData objects in iterable.

        Note: if iterable contains only one item, a copy is returned (but function
        is not applied)

        Args:
            function (Callable[[np.array, np.array], np.array]): the function to be
                applied to numpy arrays
            iterable (Sequence[VoxelData]): a sequence of VoxelData objects
        """
        iterable = list(iterable)
        if not iterable:
            raise TypeError('Attempting to reduce an empty sequence')

        for element in iterable:
            assert isinstance(element, VoxelData)
            assert_array_equal(element.voxel_dimensions, iterable[0].voxel_dimensions)
            assert_array_equal(element.offset, iterable[0].offset)

        return iterable[0].with_data(reduce(function, (x.raw for x in iterable)))


class OrientationField(VoxelData):
    """Volumetric data with rotation per voxel.

    See Also:
        https://bbpteam.epfl.ch/project/spaces/display/NRINF/Orientation+Field
    """
    def __init__(self, *args, **kwargs):
        """Init OrientationField."""
        super().__init__(*args, **kwargs)
        if self.raw.dtype not in (np.int8, np.float32, np.float64):
            raise VoxcellError(f"Invalid volumetric data dtype: {self.raw.dtype}")
        if self.payload_shape != (4,):
            raise VoxcellError("Volumetric data should store (x, y, z, w) tuple per voxel")

    # pylint: disable=arguments-differ
    def lookup(self, positions):
        """Orientations corresponding to the given positions.

        Args:
            positions: list of positions (x, y, z).

        Returns:
            Numpy array with the rotation matrices corresponding to each position.
        """
        result = super().lookup(positions, outer_value=None)

        # normalize int8 data
        if result.dtype == np.int8:
            result = result / 127.0

        # change quaternion component order: (w, x, y, z) -> (x, y, z, w)
        result = np.roll(result, -1, axis=-1)

        return quaternions_to_matrices(result)


class ROIMask(VoxelData):
    """Volumetric data defining 0/1 mask.

    See Also:
        https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?pageId=27234876
    """
    def __init__(self, *args, **kwargs):
        """Init ROIMask."""
        super().__init__(*args, **kwargs)
        if self.raw.dtype not in (np.int8, np.uint8, bool):
            raise VoxcellError(f"Invalid dtype: '{self.raw.dtype}' (expected: '(u)int8')")
        self.raw = self.raw.astype(bool)


def values_to_region_attribute(values, region_map, attr="acronym"):
    """Convert region ids to the corresponding region attribute.

    It can be used to convert the values retrieved with `VoxelData.lookup()`.

    Args:
        values (np.array): array containing the values to be converted.
        region_map (RegionMap): instance used to map values to region acronyms.
        attr (str): attribute name to lookup.

    Returns:
        Numpy array with the converted values.

    Raises:
        VoxcellError: if the attribute or any region id is not found.

    See Also:
        https://bbpteam.epfl.ch/project/spaces/display/NRINF/Scalar+Value+Image
    """
    ids, idx = np.unique(values, return_inverse=True)
    resolved = np.array([region_map.get(_id, attr=attr) for _id in ids])
    return resolved[idx]


def values_to_hemisphere(values):
    """Convert integer values 0, 1, 2 to "undefined", "right", "left" hemisphere labels.

    It can be used to convert the values retrieved with VoxelData.lookup.

    Args:
        values: numpy array containing the values to be converted.

    Returns:
        Numpy array with the converted values.

    Raises:
        VoxcellError: if any of the values is invalid.

    See Also:
        https://bbpteam.epfl.ch/project/spaces/display/NRINF/Scalar+Value+Image
    """
    ids_map = {0: "undefined", 1: "right", 2: "left"}
    ids, idx = np.unique(values, return_inverse=True)
    if not set(ids_map).issuperset(ids):
        raise VoxcellError(f"Invalid values, only {list(ids_map)} are allowed")
    resolved = np.array([ids_map[_id] for _id in ids])
    return resolved[idx]
