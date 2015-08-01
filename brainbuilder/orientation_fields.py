'''algorithm to compute orientation fields for SSCx'''

from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import vector_fields as vf
import numpy as np


def compute_sscx_orientation_fields(annotation, hierarchy, region_name):
    '''
    Accepts:
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        region_name: the exact name in the hierarchy that the field should be computed for
    Returns:
        orientation_field: volume data where every voxel contains 3 vectors: right, up, fwd
    '''
    region_mask = gb.get_regions_mask_by_names(annotation.raw, hierarchy, [region_name])

    tangents_field = vf.compute_hemispheric_spherical_tangent_fields(annotation.raw, region_mask)

    reference_mask = gb.get_regions_mask_by_ids(annotation.raw, [0])
    gradients_field = vf.calculate_fields_by_distance_to(region_mask, reference_mask)

    points_idx = np.nonzero(region_mask)

    up = vf.get_vectors_list_from_fields(gradients_field, points_idx)
    right = vf.get_vectors_list_from_fields(tangents_field, points_idx)

    fwd = np.cross(up, right)
    right = np.cross(fwd, up)

    points = gb.get_points_list_from_mask(region_mask)

    points_idx = tuple(points.transpose())
    fields = dict((name, vf.get_fields_from_vectors_list(vl,
                                                         points_idx,
                                                         annotation.mhd['DimSize']))
                  for name, vl in (('right', right), ('up', up), ('fwd', fwd)))

    return fields
