""" Algorithms to create cell positions. """

import numpy as np

from brainbuilder import poisson_disc_sampling


def _assert_cubic_voxels(voxel_data):
    '''Helper function that verifies whether the voxels of given voxel data are
    cubic.
    '''
    voxel_dims = voxel_data.voxel_dimensions
    assert voxel_dims[0] == voxel_dims[1] == voxel_dims[2]


def _get_cell_count(density, density_factor):
    '''Helper function that counts the number of cells per voxel and the total
    number of cells.
    '''
    voxel_mm3 = density.voxel_volume / 1e9  # voxel volume is in um^3
    cell_count_per_voxel = density.raw * density_factor * voxel_mm3
    cell_count = int(np.round(np.sum(cell_count_per_voxel)))
    assert cell_count > 0

    return cell_count_per_voxel, cell_count


def _get_seed(cell_count_per_voxel, voxel_data):
    '''Helper function to calculate seed for Poisson disc sampling. The seed
    is put at the centre of the voxel with the highest cell count, not to miss
    that particular point concentration
    (see also http://devmag.org.za/2009/05/03/poisson-disk-sampling/).
    '''
    idcs = np.unravel_index(np.argmax(cell_count_per_voxel),
                            cell_count_per_voxel.shape)
    return (voxel_data.indices_to_positions(idcs) +
            voxel_data.voxel_dimensions / 2.)


def _create_cell_positions_uniform(density, density_factor):
    '''Helper function that given cell density volumetric data creates cell
    positions. Within voxels, samples are created according to a uniform
    distribution.

    The total cell count is calculated based on cell density values.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all
            voxels. Default is 1.0.

    Returns:
        positions: numpy.array of shape (cell_count, 3) where each row
            represents a cell and the columns correspond to (x, y, z).
    '''
    cell_count_per_voxel, cell_count = _get_cell_count(density, density_factor)

    voxel_ijk = np.nonzero(cell_count_per_voxel > 0)
    voxel_idx = np.arange(len(voxel_ijk[0]))

    probs = 1.0 * cell_count_per_voxel[voxel_ijk] / np.sum(cell_count_per_voxel)
    chosen = np.random.choice(voxel_idx, cell_count, replace=True, p=probs)
    chosen_idx = np.stack(voxel_ijk).transpose()[chosen]

    # get random positions within chosen voxels
    return density.indices_to_positions(
        chosen_idx + np.random.random(np.shape(chosen_idx))
    )


def _create_cell_positions_poisson_disc(density, density_factor):
    '''Helper function that given cell density volumetric data creates cell
    positions with an algorithm that is based on the poisson disc sampling
    method.

    The upper limit of the total cell count is calculated based on cell density
    values. The minimum distance between points is based on the expected number
    of positions in each voxel. In case of a homogeneous cell density, the
    resulting set of points is equidistributed.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all
            voxels. Default is 1.0.

    Returns:
        positions: numpy.array of shape (nb_points, 3) where each row
            represents a cell and the columns correspond to (x, y, z). The
            upper limit of nb_points is the total cell count as extracted from
            the density volumetric data.
    '''
    cell_count_per_voxel, cell_count = _get_cell_count(density, density_factor)

    _assert_cubic_voxels(density)
    voxel_size = density.voxel_dimensions[0]

    cell_cnt_masked = np.ma.masked_values(cell_count_per_voxel, 0)
    tmp = np.divide(voxel_size, np.power(cell_cnt_masked, 1. / density.ndim))
    too_large_distance = 2 * np.max(density.bbox[1, :] - density.bbox[0, :])
    local_distance = tmp.filled(too_large_distance)
    min_distance = np.min(local_distance.flatten())

    def _min_distance_func(point=None):
        '''Helper function that makes the connection between input densities
        and distances between generated cell positions.
        '''
        if point is None:
            # minimum distance, used for the spatial index
            return min_distance
        else:
            voxel = density.positions_to_indices(point)
            return local_distance[tuple(voxel)]

    seed = _get_seed(cell_count_per_voxel, density)
    points = poisson_disc_sampling.generate_points(density.bbox, cell_count,
                                                   _min_distance_func, seed)
    return np.array(points)


def create_cell_positions(density, density_factor=1.0, method='basic'):
    '''Given cell density volumetric data, create cell positions.

    Total cell count is calculated based on cell density values.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all
            voxels. Default is 1.0.
        method: algorithm used for cell position creation. Default is 'basic'.
            - 'basic': generated positions may collide or form clusters
            - 'poisson_disc': positions are created with poisson disc sampling
                              algorithm where minimum distance between points
                              is modulated based on density values

    Returns:
        positions: numpy.array of shape (cell_count, 3) where each row represents
            a cell and the columns correspond to (x, y, z).
    '''

    position_generators = {'basic': _create_cell_positions_uniform,
                           'poisson_disc': _create_cell_positions_poisson_disc}
    return position_generators[method](density, density_factor)
