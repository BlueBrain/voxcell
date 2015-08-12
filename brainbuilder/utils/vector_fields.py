'''library to transform and handle vector fields and tensors

Note that these vector fields are represented as a list of 3D matrices with one matrix for
each dimension.

An orientation field is a list of three vector fields, corresponding to the
right, up and fwd directions (i, j, k) of a coordinate system.
'''
import h5py
import numpy as np
from scipy.ndimage import morphology

import brainbuilder.utils.genbrain as gb


####################################################################################################
# serialization

def save_orientation_fields(filename, voxel_dimensions, orientation_fields):
    '''save an orientation field to h5'''
    with h5py.File(filename, 'w') as h5:
        for name, field in orientation_fields.items():
            for i, data in enumerate(field):
                h5.create_dataset(name='%s_%d' % (name, i), data=data)

        h5.create_dataset(name='voxel_dimensions', data=voxel_dimensions)


def load_orientation_fields(filename):
    '''load an orientation field from h5'''
    orientation_fields = {}
    with h5py.File(filename, 'r') as h5:
        for k, v in h5.iteritems():
            if k != 'voxel_dimensions':
                v = np.array(v)
                name, i = k.split('_')
                i = int(i)
                orientation_fields.setdefault(name, []).append((i, v))

        voxel_dimensions = np.array(h5['voxel_dimensions'])

    for name, field in orientation_fields.items():
        field = sorted(field, key=lambda x: x[0])
        field = [f for i, f in field]
        orientation_fields[name] = field

    return voxel_dimensions, orientation_fields


####################################################################################################
# functions to create different types of fields

def _mask_fields(fields, mask):
    '''take a vector field set to (0, 0, 0) those vectors outside the given binary mask'''
    fields = [field.copy() for field in fields]

    for field in fields:
        field[~mask] = 0
    return fields


def generate_homogeneous_field(mask, direction):
    '''create an homogeneous field from a direction vector replicated according to a binary mask'''
    fields = [np.ones_like(mask, dtype=np.float32) * direction[d]
              for d in range(len(direction))]

    return _mask_fields(fields, mask)


def _calculate_fields_by_distance(target_mask, reference_mask, direction):
    '''create a vector field on target_mask where the each vector points at the closest
    voxel in reference_mask

    direction is multiplied against each vector, allowing for scaling or for
    switching the sense of the vectors'''
    distance_to_reference = morphology.distance_transform_edt(~reference_mask)
    fields = [direction * field
              for field in np.gradient(distance_to_reference)]

    return _mask_fields(fields, target_mask)


def calculate_fields_by_distance_from(region_mask, reference_mask):
    '''create a vector field on target_mask where the each vector points at the closest
    voxel in reference_mask'''
    return _calculate_fields_by_distance(region_mask, reference_mask, 1)


def calculate_fields_by_distance_to(region_mask, reference_mask):
    '''create a vector field on target_mask where the each vector points in the direction opposite
     to the closest voxel in reference_mask'''
    return _calculate_fields_by_distance(region_mask, reference_mask, -1)


def normalise_fields(fields):
    '''ensure that the vectors in a field have unit length'''
    magnitude = np.sqrt(sum(np.square(field) for field in fields))
    norm_fields = [(field / magnitude) for field in fields]
    for field in norm_fields:
        field[magnitude == 0] = 0

    return norm_fields


def get_vectors_list_from_fields(fields, voxel_indices):
    '''given the voxel_indices, return the relevant vectors from a vector field as a Nx3 matrix '''
    return np.array([field[voxel_indices] for field in fields]).transpose()


def get_fields_from_vectors_list(vectors, voxel_indices, dimensions):
    '''given an Nx3 matrix representing a list of vectors and the indices of the voxels they
    belong to, return a vector field (the rest of the voxels will contain the (0,0,0) vector)'''
    fields = [np.zeros(dimensions, dtype=np.float32) for _ in range(vectors.shape[1])]

    for i, field in enumerate(fields):
        field[voxel_indices] = vectors[:, i]

    return fields


def compute_cylindrical_tangent_field(points, center_point):
    '''create a vector field where the vectors point as tangets of a cylinder
    around the X axis'''
    # TODO make this take axis of the cylinder
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


def compute_hemispheric_spherical_tangent_fields(annotation_raw, region_mask):
    '''create a vector field as a composition of two cylindrical tangent fields, one
      for each hemisphere'''
    center_point = np.array(annotation_raw.shape) * 0.5
    center_point[2] *= 1.25
    half_region_mask = region_mask.copy()
    half_region_mask[:, :, np.arange(0, region_mask.shape[2] // 2)] = False
    points_left = gb.get_points_list_from_mask(half_region_mask)
    tangents_left = compute_cylindrical_tangent_field(points_left, center_point)

    center_point = np.array(annotation_raw.shape) * 0.5
    center_point[2] *= 0.75
    half_region_mask = region_mask.copy()
    half_region_mask[:, :, np.arange(region_mask.shape[2] // 2, region_mask.shape[2])] = False
    points_right = gb.get_points_list_from_mask(half_region_mask)
    tangents_right = compute_cylindrical_tangent_field(points_right, center_point)
    tangents_right *= -1

    tangents = np.append(tangents_left, tangents_right, axis=0)
    tangents_points = np.append(points_left, points_right, axis=0)

    tangents_field = [np.zeros_like(annotation_raw, dtype=np.float32) for _ in range(3)]
    points_idx = tuple(tangents_points.transpose())
    for i in range(3):
        tangents_field[i][points_idx] = tangents[:, i]

    return tangents_field


def combine(fields_list):
    '''take a vector field expressed as a list of 3D matrices, with one for each dimension of the
    vectors, and return instead a 4D matrix where the fourth dimension is always 3 (the vector).'''

    # TODO Change other functions to stop using a list of 3D matrices and start using 4D instead
    combined = [np.zeros_like(fields_list[0][0]),
                np.zeros_like(fields_list[0][1]),
                np.zeros_like(fields_list[0][2])]

    for i, field in enumerate(fields_list):
        already_taken = (combined[0] != 0) | (combined[1] != 0) | (combined[2] != 0)
        to_assign = (field[0] != 0) | (field[1] != 0) | (field[2] != 0)

        overwritten = np.count_nonzero(already_taken & to_assign)
        if overwritten:
            print 'field', i, 'assigning', overwritten, 'voxels already assigned'

        idx = np.nonzero(to_assign)
        combined[0][idx] = field[0][idx]
        combined[1][idx] = field[1][idx]
        combined[2][idx] = field[2][idx]

    return combined
