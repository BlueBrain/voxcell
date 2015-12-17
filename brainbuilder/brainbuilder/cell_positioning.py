'''module that positions cell bodies from a density distribution'''
import voxcell.positions


def cell_positioning(density, total_cell_count):
    '''
    Basic algorithm to create positions for cell bodies from a density distribution.

    Args:
        density: VoxelData object representing cell body density.
            Each voxel represents a value that once normalised, can be treated as a probability of
            cells appearing in this voxel.
        total_cell_count: positive int. Number of cell positions to generate.

    Returns:
        positions: numpy.array of shape (total_cell_count, 3) where each row represents
            a cell and the columns represent (x, y, z).
    '''
    return voxcell.positions.create_cell_positions(density, total_cell_count)
