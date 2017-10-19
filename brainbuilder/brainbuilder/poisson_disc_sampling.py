'''Poisson disc sampling

Credits:
- http://devmag.org.za/2009/05/03/poisson-disk-sampling/
- https://github.com/IHautaI/poisson-disc
'''

import numpy as np
from tqdm import tqdm


class Grid(object):
    '''Class representing grid, used as spatial index. Every grid point
    contains one value:
        -1 : no sample present
        x, with 0 <= x : index of sample
    '''
    def __init__(self, domain, cell_size):
        '''Constructor

        Args:
            domain: (2, dim)-numpy.array
            cell_size: scalar
        '''
        domain_size = domain[1, :] - domain[0, :]
        self.grid = np.full((domain_size / cell_size + 1).astype(int), -1)
        self.cell_size = cell_size
        self.domain = domain

    def get_grid_coords(self, point):
        '''Returns grid coordinates of point.'''
        return tuple(
            np.floor((point - self.domain[0, :]) / self.cell_size).astype(int))

    def update(self, point, index):
        '''Stores point index in grid.'''
        self.grid[self.get_grid_coords(point)] = index

    def get_sample_indices_in_neighbourhood(self, point, distance):
        '''Returns the indices of the samples that lie within a rectangular
        neighbourhood of cells. The size of the rectangle is based on an input
        distance.'''
        point_coords = np.array(self.get_grid_coords(point))

        # define neighbourhood
        nb_cells = int(np.ceil(distance / self.cell_size))
        min_corner = np.maximum(0, point_coords - nb_cells)
        max_corner = np.minimum(self.grid.shape, point_coords + nb_cells + 1)
        neighbourhood = self.grid[min_corner[0]:max_corner[0],
                                  min_corner[1]:max_corner[1],
                                  min_corner[2]:max_corner[2]]

        return neighbourhood[neighbourhood > -1]

    def no_collision(self, point, distance, sample_points):
        '''Verifies that a point does not lie closer than a given distance to
        any other existing point.

        Args:
            sample_points: list of points that are already stored on this grid
        '''
        neighbours = self.get_sample_indices_in_neighbourhood(point, distance)

        return all((np.linalg.norm(sample_points[n] - point) >= distance)
                   for n in neighbours)

    def domain_contains(self, point):
        '''Verifies whether a given point is inside the grid domain.'''
        return np.all((point >= self.domain[0, :]) & (point <= self.domain[1, :]))


def generate_point_around(point, min_distance):
    '''Generate point in spherical shell around given point at minimum distance
    from the point, according a non-uniform distribution, which favours points
    closer to the inner sphere, leading to denser packings.

    Args:
        point: three-dimensional point
        min_distance: minimum distance between point and the generated point

    Returns:
        A three-dimensional point at given minimum distance from the input
        point.
    '''
    radius = min_distance * (np.random.random() + 1)
    angle1 = 2 * np.pi * np.random.random()
    angle2 = 2 * np.pi * np.random.random()

    d_x = np.cos(angle1) * np.sin(angle2)
    d_y = np.sin(angle1) * np.sin(angle2)
    d_z = np.cos(angle2)
    return point + radius * np.array([d_x, d_y, d_z])


def _get_seed(domain):
    '''Helper function that generates random seed according to a uniform
    distribution over a given domain.'''
    domain_size = domain[1, :] - domain[0, :]
    return domain[0, :] + np.random.random(domain[0, :].shape) * domain_size


def generate_points(domain, nb_points, min_distance, seed=None,
                    nb_trials=30, display_progress=True):
    '''Generate a number of points with Poisson disc sampling.

    Args:
        domain: (2, dim)-numpy.array
        nb_points: number of desired points
        min_distance: a function that returns the minimum distance between two
                      points based on point-coordinates. If not coordinates are
                      passed, the absolute minimum should be returned.
        seed: first sample point (numpy.array)
        nb_trials: number of trials each time a new point is generated
        display_progress: boolean that indicates whether a progress bar is
                          displayed. Default is True.

    Returns:
        A list of points.
    '''
    # initialisation of helper containers
    grid = Grid(domain, min_distance() / np.sqrt(domain.shape[1]))
    active_list = []
    sample_points = []

    def _add_to_containers(point):
        '''Update containers used for Poisson disc sampling.'''
        sample_points.append(point)
        idx = len(sample_points) - 1
        active_list.append(idx)
        grid.update(point, idx)

    # first point is seed point
    if seed is None:
        seed = _get_seed(domain)
    if grid.domain_contains(seed):
        _add_to_containers(seed)

    # generate points
    if display_progress:
        progress_bar = tqdm(total=nb_points)
        progress_bar.update(1) # count the seed as the first

    while active_list and (len(sample_points) < nb_points):
        idx = np.random.choice(active_list)
        point = sample_points[idx]
        active_list.remove(idx)

        for _ in range(nb_trials):
            new_pt = generate_point_around(point, min_distance(point))

            if grid.domain_contains(new_pt) and grid.no_collision(
                    new_pt, min_distance(new_pt), sample_points):
                _add_to_containers(new_pt)
                if display_progress:
                    progress_bar.update(1)

            if len(sample_points) == nb_points:
                break

    if display_progress:
        progress_bar.close()

    return sample_points
