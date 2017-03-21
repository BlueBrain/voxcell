'''algorithm to compute orientation fields for SSCx'''
from voxcell import build
from voxcell import vector_fields as vf

import numpy as np


def compute_orientation_field(annotation, hierarchy, region_name):
    '''Computes the orientation field for the somatosensory cortex

    Args:
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        region_name: the exact name in the hierarchy that the field should be computed for

    Returns:
        A 5D numpy array of shape AxBxCx3x3 where AxBxC is the shape of annotation, the first
        dimension of size 3 differentiates between the right,up,forwards fields and the last
        dimension of size 3 contains the three i,j,k components of each vector

    '''
    region_mask = build.mask_by_region_names(annotation.raw, hierarchy, [region_name])

    right_field = vf.compute_hemispheric_spherical_tangent_fields(annotation.raw, region_mask)

    reference_mask = build.mask_by_region_ids(annotation.raw, [0])
    up_field = vf.calculate_fields_by_distance_to(region_mask, reference_mask)

    fwd_field = np.cross(up_field, right_field)
    right_field = np.cross(fwd_field, up_field)

    field = vf.combine_vector_fields([right_field, up_field, fwd_field])

    return annotation.with_data(field)
