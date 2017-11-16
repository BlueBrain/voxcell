'''See https://bbpteam.epfl.ch/project/issues/browse/BRBLD-52'''

from __future__ import print_function

import sys
import csv
import multiprocessing
import argparse

from scipy import optimize

import numpy as np
import voxcell.voxel_data as vd
import brainbuilder.poisson_disc_sampling as poisson_disc
import matplotlib.pyplot as plt


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser("Filter extneurondb.dat")
    parser.add_argument(
        "--start",
        help="Start density (# points / mm3)",
        type=float,
        required=True
    )
    parser.add_argument(
        "--stop",
        help="Stop density (# points / mm3)",
        type=float,
        required=True
    )
    parser.add_argument(
        "--nb_steps",
        help="Number of density values tested",
        type=int,
        default=2,
        required=False
    )
    parser.add_argument(
        "--output",
        help="File name used for output files",
        default="minimum_distance_vs_nb_points",
        required=False
    )
    result = parser.parse_args()

    if result.start > result.stop:
        raise Exception('Stop density must be greater then or equal to start density.')

    return result


def create_density_profile(desired_nb_points):
    '''Create density profile with dimensions so that total cell count is
    desired_nb_points.'''
    nb_voxels = 100
    voxel_size = 10 # length unit is um
    data = np.full((nb_voxels, nb_voxels, nb_voxels), desired_nb_points)
    return vd.VoxelData(data, (voxel_size, voxel_size, voxel_size))


def error_nb_points(distance, bbox, desired_nb_points):
    '''Returns absolute difference between desired number of points and number
    of points created with poisson disk sampling for given minimum distance.
    Samples are generated within a given bounding box.
    '''
    def _min_distance_func(point=None):
        '''Helper function that makes the connection between input densities
        and distances between generated cell positions.
        '''
        if point is None:
            return distance # for the creation of the spatial grid
        return distance

    points = poisson_disc.generate_points(
        bbox, sys.maxsize, _min_distance_func, display_progress=False)

    return np.abs(len(points) - desired_nb_points)


def find_input_minimum_distance(desired_nb_points):
    '''Find minimum distance for which the poisson disk sampling method obtains
    the desired number of sample points.
    '''
    density = create_density_profile(desired_nb_points)

    # assuming cubic domain and homogeneous data
    initial_guess = np.divide(0.84 * (density.bbox[1, 0] - density.bbox[0, 0]),
                              np.power(desired_nb_points, 1. / density.ndim))

    result = optimize.minimize(
        error_nb_points, initial_guess, method='Nelder-Mead',
        args=(density.bbox, desired_nb_points),
        options={'xatol': 0.001, 'fatol': 0.001})

    print('{} points, min distance = {}'.format(desired_nb_points, result.x[0]))
    return desired_nb_points, result.x[0]


def fit_power_law(xdata, ydata):
    '''Power-law fitting. We assume a relation of
            y = a * x^b
    and fit
            log(y) = log(a) + b*log(x)
    '''
    def _fitfunc(p, x):
        return p[0] + p[1] * x

    def _errfunc(p, x, y):
        return y - _fitfunc(p, x)

    initial_guess = [0., 0.]
    logx = np.log10(xdata)
    logy = np.log10(ydata)

    out = optimize.leastsq(_errfunc, initial_guess, args=(logx, logy),
                           full_output=1)
    coeff = 10**out[0][0]
    exponent = out[0][1]
    print('Fitting result: D = {} * N ^ {}'.format(coeff, exponent))
    return coeff, exponent


def save_data(output_path, xdata, ydata):
    '''Save simulation data in two columns.'''
    with open(output_path, 'wb') as output_file:
        writer = csv.writer(output_file, delimiter=' ')
        writer.writerows(zip(xdata, ydata))
    print('Output data written to {}'.format(output_path))


def save_plot_data_and_power_law(output_path, xdata, ydata, coeff, exponent):
    '''Create and save figure with data points and fitted curve.'''
    plt.subplot(2, 1, 1)
    power_law = 'D = {:10.3f} N^{:1.3f}'.format(coeff, exponent)
    plt.plot(xdata, coeff * np.power(xdata, exponent), label=power_law)
    plt.plot(xdata, ydata, 'x', label='Data points')
    plt.title('Minimum distance required to obtain desired number of points')
    plt.xlabel('Number of points N')
    plt.ylabel('Minimum distance D')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, coeff * np.power(xdata, exponent), label=power_law)
    plt.loglog(xdata, ydata, 'x', label='Data points')
    plt.xlabel('Number of points N (log scale)')
    plt.ylabel('Minimum distance D (log scale)')
    plt.legend()

    plt.tight_layout()
    plt.gcf().savefig(output_path)
    print('Output figure written to {}'.format(output_path))


if __name__ == "__main__":
    args = parse_args()

    # test range of number of points
    nb_points = np.linspace(args.start, args.stop, args.nb_steps)

    # calculate minimum distance that recreates the exact number of points
    pool = multiprocessing.Pool()
    results = pool.map(find_input_minimum_distance, nb_points)
    nb_points, min_distance = zip(*results)

    # save data for later reuse
    output_data_path = "{}.txt".format(args.output)
    save_data(output_data_path, nb_points, min_distance)

    # fit result
    fit_coeff, fit_exponent = fit_power_law(nb_points, min_distance)

    # plot minimum distance as function of number of sample points
    output_figure_path = "{}.png".format(args.output)
    save_plot_data_and_power_law(output_figure_path, nb_points, min_distance,
                                 fit_coeff, fit_exponent)
