'''algorithm to assign orientations to a group of cells'''


from brainbuilder.utils import genbrain as gb
import numpy as np


def assign_orientations(positions, orientation_fields, voxel_dimensions):
    '''
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        orientation_field: volume data where every voxel contains 3 vectors: right, up, fwd
    Returns:
        orientations: list of orientations (3 vectors: right, up, fwd).
            The order matches that of the positions list.
    '''
    voxel_idx = gb.cell_positions_to_voxel_indices(positions, voxel_dimensions)
    idx = tuple(voxel_idx.transpose())

    vectors = []
    for name in ('right', 'up', 'fwd'):
        field = orientation_fields[name]
        vectors.append(np.array([d[idx] for d in field]).transpose())

    return vectors


# pylint: disable=W0613
def randomise_orientations(orientations, ranges):
    '''
    Takes some orientations and applies random rotations to them according to specific ranges.
    This allows us to define regions where orientation is strict only to a certain degree and
    in certain dimensions (for example: sccx cares about the XZ plane,
    but fully random rotations around Y may be desirable).

    Accepts:
        orientations: list of orientations (3 vectors: right, up, fwd).
        rotation_ranges: list of angle ranges around the three main axis: X, Y, Z
    Returns:
        orientations: list of orientations (3 vectors: right, up, fwd).
    '''
    # TODO do
    return orientations
