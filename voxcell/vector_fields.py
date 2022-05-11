"""Library to build, transform and handle fields of vectors and orientations.

A vector field is a volumetric dataset that expresses a 3D vector for each voxel.
Vector fields are represented as 4D numpy arrays. The first three dimensions of
the array represent space, the last dimension is always of size 3 and contains the three components
(i, j, k) of the vector for that voxel.


An orientation field is a volumetric dataset that expresses a rotation matrix for each voxel.
Orientation fields are represented as 5D numpy arrays. The first three dimensions of the array
represent space, the fourth and fifth dimensions contain a 3x3 rotation matrix.

Note that although 3D is our main target, most of these functions should behave correctly for lower
and higher dimensionality levels.
"""

import logging

import numpy as np
import scipy.ndimage

L = logging.getLogger(__name__)


def gaussian_filter(vf, sigma):
    """Apply a gaussian filter to a vector field.

    Note that filter is applied without normalizing the input or output.

    Args:
        vf: Vector field
        sigma : scalar. Standard deviation for Gaussian kernel.

    Returns:
        The resulting vector field not normalized.
    """
    filtered = vf.copy()
    mask = np.any(vf != 0, axis=-1)

    filtered[..., 0] = mask * scipy.ndimage.gaussian_filter(filtered[..., 0], sigma=sigma)
    filtered[..., 1] = mask * scipy.ndimage.gaussian_filter(filtered[..., 1], sigma=sigma)
    filtered[..., 2] = mask * scipy.ndimage.gaussian_filter(filtered[..., 2], sigma=sigma)

    return filtered


def combine_vector_fields(fields):
    """Given a list of vector fields return an orientation field.

    The vectors from the fields are treated as a new base and will be stored as column vectors so
    that the matrices of the resulting field (one per voxel) can be used to premultiply vectors
    for rotation.

    Args:
        fields: a list of vector fields.
            All of the fields are expected to have the same shape (AxBxCx3)

    Returns:
        A 5D numpy array representing an orientation field
    """
    if not fields:
        return np.empty((0,))

    shape = fields[0].shape

    # add a second-to-last dimension: the number of fields
    result = np.zeros(shape=shape[:-1] + (len(fields),) + (shape[-1],), dtype=fields[0].dtype)

    # abusing numpy broadcasting here saves us having to do an explicit transpose afterwards
    for i, f in enumerate(fields):
        result[..., i] = f

    return result


def split_orientation_field(field):
    """Given an orientation field return a list of vector fields."""
    return [field[..., i] for i in range(field.shape[-2])]


def join_vector_fields(vf0, *vfs):
    """Performs an union of several vector fields.

    A voxel on a vector field is considered "empty" if all of its components are zero.

    Args:
        vf0: first vector field.
        vfs: rest of vector fields. All of them must have the same shape.
    """
    vfs = (vf0,) + vfs
    assert all(v.shape == vf0.shape for v in vfs)  # pylint: disable=no-member

    joined = np.zeros_like(vf0)
    joined_mask = np.zeros(joined.shape[:-1],  # pylint: disable=unsubscriptable-object
                           dtype=bool)
    for field in vfs:
        field_mask = np.any(field != 0, axis=-1)
        overlap_count = np.count_nonzero(joined_mask & field_mask)
        if overlap_count:
            L.warning('%d voxels overlap will be overwritten', overlap_count)
        joined_mask |= field_mask
        joined[field_mask] = field[field_mask]

    return joined
