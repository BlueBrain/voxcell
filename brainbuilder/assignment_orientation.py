'''algorithm to assign orientations to a group of cells'''
from brainbuilder.utils import genbrain as gb


def assign_orientations(positions, orientation_fields, voxel_dimensions):
    '''Assigns orientations to each of the positions based on the orientation_field

    Args:
        positions: list of positions for soma centers (x, y, z).
        orientation_field: volume data where every voxel contains 3 vectors: right, up, fwd
        voxel_dimensions: tuple with the size of the voxels in microns in each axis.

    Returns:
        orientations: a Nx3x3 numpy array where N is the number of positions and the 3x3 are
            rotation matrices.
    '''
    voxel_idx = gb.cell_positions_to_voxel_indices(positions, voxel_dimensions)
    idx = tuple(voxel_idx.transpose())
    return orientation_fields[idx]


# pylint: disable=W0613
def randomise_orientations(orientations, ranges):
    '''
    Takes some orientations and applies random rotations to them according to specific ranges.
    This allows us to define regions where orientation is strict only to a certain degree and
    in certain dimensions (for example: sccx cares about the XZ plane,
    but fully random rotations around Y may be desirable).

    Args:
        orientations: list of orientations (3 vectors: right, up, fwd).
        rotation_ranges: list of angle ranges around the three main axis: X, Y, Z

    Returns:
        orientations: list of orientations (3 vectors: right, up, fwd).
    '''
    # TODO do
    return orientations
