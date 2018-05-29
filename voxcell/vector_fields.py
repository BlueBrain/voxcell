'''library to build, transform and handle fields of vectors and orientations

A vector field is a volumetric dataset that expresses a 3D vector for each voxel.
Vector fields are represented as 4D numpy arrays. The first three dimensions of
the array represent space, the last dimension is always of size 3 and contains the three components
(i, j, k) of the vector for that voxel.


An orientation field is a volumetric dataset that expresses a rotation matrix for each voxel.
Orientation fields are represented as 5D numpy arrays. The first three dimensions of the array
represent space, the fourth and fifth dimensions contain a 3x3 rotation matrix.

Note that although 3D is our main target, most of these functions should behave correctly for lower
and higher dimensionality levels.
'''

import logging

import numpy as np
from scipy.ndimage import morphology
import scipy.ndimage

from voxcell.utils import deprecate


L = logging.getLogger(__name__)


def generate_homogeneous_field(mask, direction):
    '''create an homogeneous field from a direction vector replicated according to a binary mask'''
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    field = np.zeros(shape=(mask.shape + direction.shape), dtype=direction.dtype)
    field[mask] = direction
    return field


def _calculate_fields_by_distance(target_mask, reference_mask, direction):
    '''create a vector field on target_mask where the each vector points at the closest
    voxel in reference_mask

    direction is multiplied against each vector, allowing for scaling or for
    switching the sense of the vectors

    Returns:
        A 4D numpy array with the same shape as reference_mask but with an extra dimension
        of size 3 that contains the vectors for each voxel.
    '''
    distance_to_reference = morphology.distance_transform_edt(~reference_mask)

    result = np.zeros(shape=(reference_mask.shape + (len(reference_mask.shape),)), dtype=np.float32)

    gradient = np.gradient(distance_to_reference)

    if not isinstance(gradient, (tuple, list)):
        # when working on 1D numpy return is inconsistent
        gradient = (gradient,)

    for i, field in enumerate(gradient):
        result[..., i] = direction * target_mask * field

    return result


def calculate_fields_by_distance_from(region, reference):
    '''create a vector field on target_mask where the each vector points at the closest
    voxel in reference_mask
    If reference is part of region, the vectors there will be (0,0,0).
    Result is not normalized.'''
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    return _calculate_fields_by_distance(region, reference, 1)


def calculate_fields_by_distance_to(region, reference):
    '''create a vector field on target_mask where the each vector points in the direction opposite
    to the closest voxel in reference_mask.
    If reference is part of region, the vectors there will be (0,0,0).
    Result is not normalized.'''
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    return _calculate_fields_by_distance(region, reference, -1)


def calculate_fields_by_distance_between(region, first, last):
    '''create a vector field on region where the vector in each voxel points in the direction
    from first to last.
    If first and last overlap, the vectors there will be (0,0,0).
    Result is not normalized.'''
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    field_to_last = calculate_fields_by_distance_to(region, last)
    field_from_first = calculate_fields_by_distance_from(region, first)
    return join_vector_fields(field_to_last, field_from_first)


def compute_cylindrical_tangent_vectors(points, center_point):
    '''create a vector for each of the points that point as tangents of a cylinder
    around the X axis

    Returns:
        A numpy array of the same shape as points (Nx3)
    '''
    # TODO make this take axis of the cylinder
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    from_center = points - center_point
    tangents = np.zeros_like(from_center)

    not_zero_x = from_center[:, 1] != 0
    tangents[not_zero_x, 1] = -(from_center[not_zero_x, 2] * 1) / from_center[not_zero_x, 1]
    tangents[not_zero_x, 2] = 1

    not_zero_y = (from_center[:, 1] == 0) & (from_center[:, 2] != 0)
    tangents[not_zero_y, 1] = 1
    tangents[not_zero_y, 2] = -(from_center[not_zero_y, 1] * 1) / from_center[not_zero_y, 2]

    not_zero_z = (from_center[:, 1] == 0) & (from_center[:, 2] == 0)
    tangents[not_zero_z] = np.array([0, 1, 0])

    assert not np.any(np.isnan(tangents))

    tangents_lengths = np.sqrt(np.sum(np.square(tangents), axis=1))
    tangents_norm = tangents / tangents_lengths[:, np.newaxis]

    return tangents_norm


def _get_points_list_from_mask(mask):
    '''get all the voxel indices for positive values of the binary voxel mask
    for example: for a mask that has everything to true, the result will look like:
    [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 0],
     [0, 1, 1],
           ...]
    '''
    return np.array(np.nonzero(mask)).swapaxes(0, 1)


def compute_hemispheric_spherical_tangent_fields(annotation_raw, region_mask):
    '''create a vector field as a composition of two cylindrical tangent fields, one
    for each hemisphere

    Returns:
        A 4D numpy array representing a vector field with a shape equivalent to annotation_raw.
    '''
    deprecate.warn("""
        This method would be deprecated in voxcell==3.0.
        Please contact NSE team if you are using it.
    """)
    center_point = np.array(annotation_raw.shape) * 0.5
    center_point[2] *= 1.25
    half_region_mask = region_mask.copy()
    half_region_mask[:, :, np.arange(0, region_mask.shape[2] // 2)] = False
    points_left = _get_points_list_from_mask(half_region_mask)
    tangents_left = compute_cylindrical_tangent_vectors(points_left, center_point)

    center_point = np.array(annotation_raw.shape) * 0.5
    center_point[2] *= 0.75
    half_region_mask = region_mask.copy()
    half_region_mask[:, :, np.arange(region_mask.shape[2] // 2, region_mask.shape[2])] = False
    points_right = _get_points_list_from_mask(half_region_mask)
    tangents_right = compute_cylindrical_tangent_vectors(points_right, center_point)
    tangents_right *= -1

    tangents = np.append(tangents_left, tangents_right, axis=0)
    tangents_points = np.append(points_left, points_right, axis=0)

    tangents_field = np.zeros(shape=(annotation_raw.shape + (tangents.shape[1],)), dtype=np.float32)
    points_idx = tuple(tangents_points.transpose())
    tangents_field[points_idx] = tangents

    return tangents_field


def normalize(vf):
    '''normalize a vector field'''
    from voxcell.math_utils import normalize as _normalize
    deprecate.warn(
        "`voxcell.vector_fields.normalize()` method has been moved to `voxcell.math_utils`"
    )
    return _normalize(vf)


def gaussian_filter(vf, sigma):
    '''apply a gaussian filter to a vector field

    Note that filter is applied without normalizing the input or output.

    Args:
        vf: Vector field
        sigma : scalar. Standard deviation for Gaussian kernel.

    Returns:
        The resulting vector field not normalized.
    '''
    filtered = vf.copy()
    mask = np.any(vf != 0, axis=-1)

    filtered[..., 0] = mask * scipy.ndimage.gaussian_filter(filtered[..., 0], sigma=sigma)
    filtered[..., 1] = mask * scipy.ndimage.gaussian_filter(filtered[..., 1], sigma=sigma)
    filtered[..., 2] = mask * scipy.ndimage.gaussian_filter(filtered[..., 2], sigma=sigma)

    return filtered


def combine_vector_fields(fields):
    '''given a list of vector fields return an orientation field

    The vectors from the fields are treated as a new base and will be stored as column vectors so
    that the matrices of the resulting field (one per voxel) can be used to premultiply vectors
    for rotation.

    Args:
        fields: a list of vector fields.
            All of the fields are expected to have the same shape (AxBxCx3)

    Returns:
        A 5D numpy array representing an orientation field
    '''

    if fields:
        shape = fields[0].shape

        # add a second-to-last dimension: the number of fields
        result = np.zeros(shape=shape[:-1] + (len(fields),) + (shape[-1],), dtype=fields[0].dtype)

        # abusing numpy broadcasting here saves us having to do an explicit transpose afterwards
        for i, f in enumerate(fields):
            result[..., i] = f

        return result

    else:
        return np.empty((0,))


def split_orientation_field(field):
    '''given an orientation field return a list of vector fields'''
    return [field[..., i] for i in range(field.shape[-2])]


def join_vector_fields(vf0, *vfs):
    '''performs an union of several vector fields.
    A voxel on a vector field is considered "empty" if all of its components are zero.

    Args:
        vf0: first vector field.
        vfs: rest of vector fields. All of them must have the same shape.
    '''
    vfs = (vf0,) + vfs
    assert all(v.shape == vf0.shape for v in vfs)  # pylint: disable=no-member

    joined = np.zeros_like(vf0)
    joined_mask = np.zeros(joined.shape[:-1], dtype=np.bool)

    for field in vfs:
        field_mask = np.any(field != 0, axis=-1)
        overlap_count = np.count_nonzero(joined_mask & field_mask)
        if overlap_count:
            L.warning('%d voxels overlap will be overwritten', overlap_count)
        joined_mask |= field_mask
        joined[field_mask] = field[field_mask]

    return joined
