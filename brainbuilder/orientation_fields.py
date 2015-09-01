'''algorithm to compute orientation fields for SSCx'''
import h5py

from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import vector_fields as vf
import numpy as np


VECTOR_NAMES = ('right', 'up', 'fwd', )


def compute_sscx_orientation_fields(annotation, hierarchy, region_name):
    '''Computes the orientation field for the somatosensory cortex

    Args:
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


def serialize_orientation_fields(dst_file, orientation_fields):
    '''Serialize orientation fields

    Args:
        dst_file(str): fullpath to filename to write
        orientation_fields: dict w/ keys ('fwd', 'right', 'up') -> np.array
    '''
    assert set(orientation_fields.keys()) == set(VECTOR_NAMES)

    with h5py.File(dst_file, 'w') as h5:
        for name, orientations in orientation_fields.iteritems():
            #change to 4D array
            #TODO this is very inefficient, everything should really be 4D already
            data = np.array((orientations[0],
                             orientations[1],
                             orientations[2]))
            h5.create_dataset(name, data=data)


def deserialize_orientation_fields(src_file):
    '''De-serialize orientation fields

    Args:
        src_file(str): fullpath to filename to write

    Returns:
        orientation_fields: dict w/ keys ('fwd', 'right', 'up') -> np.array
    '''
    fields = {}
    with h5py.File(src_file, 'r') as h5:
        for name in VECTOR_NAMES:
            #change from 4D array
            fields[name] = [np.array(h5[name][0]),
                            np.array(h5[name][1]),
                            np.array(h5[name][2]),
                            ]

    return fields
