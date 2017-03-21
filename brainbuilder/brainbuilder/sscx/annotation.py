""" Build atlas for SSCX column based on BBP recipe. """

import voxcell.build as vb
from voxcell.core import VoxelData, Hierarchy

from brainbuilder.utils import bbp

import numpy as np


def build_column_atlas(recipe_filename):
    """ Build atlas for SSCX column based on BBP recipe. """

    lattice_vectors = bbp.get_lattice_vectors(recipe_filename)
    hexagon_side = np.linalg.norm(lattice_vectors['a1'])
    voxel_x = voxel_z = hexagon_side / 10
    hexagon_mask = vb.regular_convex_polygon_mask_from_side(hexagon_side, 6, voxel_x)

    layer_thickness_microns = bbp.get_layer_thickness(recipe_filename)

    # We want our atlas to contain a bit of space tagged as "outside the brain".
    # This will later be used to compute "distance to pia" for placement hints.
    layer_thickness_microns[0] = 15  # PIA
    layer_ids, heights_microns = zip(*sorted(layer_thickness_microns.items(), lambda l, _: -l[0]))
    voxel_y = vb.get_voxel_side(heights_microns)

    heights = np.round(np.array(heights_microns) / voxel_y).astype(np.uint)
    annotation_data = vb.layered_annotation(hexagon_mask.shape, heights, layer_ids)

    column_mask = vb.column_mask(hexagon_mask, length=annotation_data.shape[1], axis=1)
    annotation_data[~column_mask] = 0

    annotation = VoxelData(annotation_data, (voxel_x, voxel_y, voxel_z))

    hierarchy = Hierarchy({
        'name': 'root',
        'id': 999,
        'children': [
            {'name': 'Pia', 'id': 0},
            {'name': 'Layer 1', 'id': 1},
            {'name': 'Layer 2', 'id': 2},
            {'name': 'Layer 3', 'id': 3},
            {'name': 'Layer 4', 'id': 4},
            {'name': 'Layer 5', 'id': 5},
            {'name': 'Layer 6', 'id': 6},
        ]
    })

    return annotation, hierarchy
