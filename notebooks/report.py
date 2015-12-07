'''a collection of very ad hoc tests and plots to analyse the rebuilt BBP microcircuit'''
import itertools
import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from voxcell import math


def count_cell_percentages(recipe, cells, attributes):
    '''compute cell percentages, actual and expected, grouped by the given attributes'''
    actual = []
    expected = []
    total_cell_count = cells.positions.shape[0]

    for values, rows in recipe.groupby(attributes):
        expected.append(rows.percentage.sum())

        values = pd.Series(values, index=attributes)
        cell_count = np.count_nonzero(np.all(values == cells.properties[attributes], axis=1))
        actual.append(float(cell_count) / total_cell_count)

    return np.array(actual), np.array(expected)


def report_cell_percentages(recipe, columns, attributes):
    '''plot cell percentages, actual and expected, grouped by the given attributes'''
    actual, expected = zip(*[count_cell_percentages(recipe, c, attributes) for c in columns])

    plt.figure(figsize=(15, 2))
    plt.title('Percentages of cells')
    plt.xlabel('recipe row (grouped by %s)' % ', '.join(attributes))
    plt.ylabel('cell count (% of total)')
    for e in expected:
        plt.plot(e, 'b:')
        plt.plot(e, 'b+', label='expected')
    for a in actual:
        plt.plot(a, 'rx', label='actual')
    plt.legend(loc='upper left')

    plt.figure(figsize=(15, 2))
    plt.title('Percentages of cells (Zoom in for < 0.1%)')
    plt.xlabel('recipe row (grouped by %s)' % ', '.join(attributes))
    plt.ylabel('cell count (% of total)')
    plt.ylim((0, 0.001))
    for e in expected:
        plt.plot(e, 'b+')
    for a in actual:
        plt.plot(a, 'rx')

    plt.figure(figsize=(15, 2))
    plt.title('Absolute error in percentages of cells')
    plt.xlabel('recipe row (grouped by %s)' % ', '.join(attributes))
    plt.ylabel('error (% difference)')
    for a, e in zip(actual, expected):
        plt.plot(np.abs(e - a), 'r:')
        plt.plot(np.abs(e - a), 'ro')


def count_morphology_used(neurondb, cells, attributes):
    '''count how many unique morphologies where used and available for
    cells grouped by the given attributes'''
    keys = []
    usage = []
    availability = []

    for values, rows in cells.properties.groupby(attributes):
        used = rows.morphology.unique()

        values = pd.Series(values, index=attributes)
        matching_entries = np.all(values == neurondb[attributes], axis=1)
        available = neurondb[matching_entries].morphology.unique()

        keys.append(values)
        usage.append(len(used))
        availability.append(len(available))

    return keys, np.array(usage), np.array(availability)


def report_morphology_used(neurondb, column):
    '''plot how many unique morphologies where used and available for
    cells grouped by the given attributes'''
    attributes = ['mtype', 'etype']
    keys, u, a = count_morphology_used(neurondb, column, attributes)

    plt.figure(figsize=(15, 50))
    plt.title('Morphology Usage')
    plt.barh(np.arange(len(a)), a, label='available')
    plt.barh(np.arange(len(u)), u, color='r', label='used')
    plt.xlabel('number of different morphologies')
    plt.ylabel('morphologies grouped by %s' % ', '.join(attributes))

    labels = ['%s - %s' % tuple(k.values) for k in keys]
    plt.yticks(np.arange(len(labels)) + .5, labels)

    plt.ylim(0, a.shape[0])
    plt.legend()


def report_morphology_y_scatter(column):
    '''create a scatter plot of used unique morphologies representing each soma height as a dot'''
    for lid in column.properties.layer.unique():
        mask = (column.properties.layer == lid).values
        unique_morphologies, morphology_indices = np.unique(column.properties[mask].morphology,
                                                            return_inverse=True)
        plt.figure(figsize=(15, 2))
        plt.title('Cell Scatter Plot - Layer %d (%d cells)' %
                  (lid + 1, column.positions[mask, :].shape[0]))
        plt.xlabel('morphologies')
        plt.ylabel('height (microns)')
        plt.scatter(morphology_indices, column.positions[mask, 1], marker=',', s=1, lw = 0)
        plt.xlim(0, len(unique_morphologies))


def report_height_histogram(columns):
    '''plot a height histogram for each layer'''
    layer_colors_light = {0: '#8700e5', 1: '#fdff00', 2: '#b0ff00',
                          3: '#00ecff', 4: '#00db25', 5: '#ff65ca'}

    layer_colors_dark = {0: '#510089', 1: '#c9af13', 2: '#8ea72f',
                         3: '#0296ab', 4: '#008917', 5: '#a3006a'}

    for lid in layer_colors_dark:
        plt.figure()
        plt.title('Layer %d' % (lid + 1))
        plt.xlabel('bin centre (microns)')
        plt.ylabel('cell count')

        all_bincenters = []
        all_counts = []
        for c in columns:
            y = c.positions[(c.properties.layer == lid).values, 1]
            counts, bins = np.histogram(y, bins=20)
            bincenters = 0.5 * (bins[1:] + bins[:-1])
            all_bincenters.append(bincenters)
            all_counts.append(counts)
            plt.plot(bincenters, counts, color=layer_colors_light[lid], linestyle='dotted')

        plt.plot(np.mean(all_bincenters, axis=0), np.mean(all_counts, axis=0),
                 color=layer_colors_dark[lid])


def report_tiling(columns, hexagon_side):
    '''Plot a cell density histogram of a stripe on the Z axis of every pair of two hexagons
    one right above the other (so their flat sides touch)'''
    nbins = 25
    bins = np.arange(1.5 * nbins + 1) * (hexagon_side * 3.) / nbins - hexagon_side * .5
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    stripe_width = 200
    hexagon_height = 2 * hexagon_side * np.sin(2 * np.pi / 6)

    plt.figure()
    all_counts = []
    for c1, c2 in itertools.product(columns, columns):
        meanx1 = np.mean(c1.positions[:, 0])
        stripe1 = np.abs(c1.positions[:, 0] - meanx1) < stripe_width * 0.5
        meanx2 = np.mean(c2.positions[:, 0])
        stripe2 = np.abs(c2.positions[:, 0] - meanx2) < stripe_width * 0.5
        counts, _ = np.histogram(np.append(c2.positions[stripe2, 2],
                                           c1.positions[stripe1, 2] + hexagon_height), bins=bins)
        all_counts.append(counts)
        plt.plot(bincenters, counts, linestyle='dotted', color='lightblue')

    av_counts = np.average(all_counts, axis=0)
    plt.plot(bincenters, av_counts, '-')
    plt.plot([0, 0], [0, 100], 'r-')
    plt.plot([hexagon_height, hexagon_height], [0, 100], 'r-')
    plt.plot([hexagon_height * 2, hexagon_height * 2], [0, 100], 'r-')

    plt.title('density in a %d micron-wide stripe on the Z axis' % stripe_width)
    plt.ylabel('cell count')
    plt.xlabel('z bin centre (microns)')


def check_hexagon_diameter(columns, hexagon_side):
    '''print warnings if any column goes outside their hexagon diameter'''
    for i, c in enumerate(columns):
        hexagon_diameter = hexagon_side * 2.
        aabb_min, aabb_max = math.positions_minimum_aabb(c.positions)
        excess = (aabb_max - aabb_min) - hexagon_diameter
        if excess[0] > 0:
            print ('Column %d: X component is %f (%f%%) microns bigger than hexagon diameter' %
                   (i, excess[0], 100 * excess[0] / hexagon_diameter))
        if excess[2] > 0:
            print ('Column %d: Z component is %f (%f%%) microns bigger than hexagon diameter' %
                   (i, excess[2], 100 * excess[2] / hexagon_diameter))


def collect_placement_hints(neurondb):
    '''return a dict mapping tuples (layer, mtype, etype) that identify groups of morphologies,
     to an array representing the sum of the placement hints of all of those morphologies'''
    hints = {}

    for lid, layerdf in neurondb.groupby('layer'):
        for key, mtypedf in layerdf.groupby(['layer', 'mtype', 'etype']):
            phs = list(mtypedf.placement_hints.values)
            ph_lengths = np.array([len(l) for l in phs])
            if np.all(ph_lengths == ph_lengths[0]):
                hints[key] = np.sum(phs, axis=0)

            else:
                # this actually never happens with the current neurondb
                logging.error('Ignoring %s because has different placement hint lengths: %s',
                              (key[0], key[1], key[2],
                               ', '.join(str(phl) for phl in np.unique(ph_lengths))))

    return hints


def check_bad_bins(hints, column):
    '''confirm that any cell missing a morphology belongs to a group for which there are
    sections of the layer with a placement hint score of zero'''
    with_bad_bins = [k for k, v in hints.iteritems() if np.any(v == 0)]

    no_morph = column.properties.morphology.isnull()
    for k in column.properties[no_morph][['layer', 'mtype', 'etype']].drop_duplicates().values:
        if tuple(k) not in with_bad_bins:
            print ('Unexpected group %s %s %s does not have any bad bin. '
                   'It should have got a morphology' % tuple(k))


def report_placement_hints(hints):
    '''plot placement hints for groups of morphologies and
    warn about groups with bins of zero score'''
    classification = {}
    for k, summed in hints.iteritems():
        classification.setdefault(k[0], {}).setdefault(len(summed), []).append((k, summed))

    for lid, by_ph_length in classification.iteritems():
        for phlength, data in by_ph_length.iteritems():
            plt.figure(figsize=(15, 3))
            plt.title('Placement Hints')
            plt.xlabel('layer height bin number')
            plt.ylabel('score')
            for key, phs in data:
                line = plt.plot(phs, ':')
                color = line[0].get_color()
                plt.plot(phs, 'o', color=color, label='%s %s %s' % key)
                if np.any(phs == 0):
                    print ('%s %s %s contains %d bins with score zero' %
                           (key + (np.count_nonzero(phs == 0),)))

            # the legend gets huge and unreadable
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.figure(figsize=(15, 3))
            plt.title('Placement Hints (ZOOMED)')
            plt.xlabel('layer height bin number')
            plt.ylabel('score')
            plt.ylim(-1, 20)
            for key, phs in data:
                line = plt.plot(phs, ':')
                color = line[0].get_color()
                plt.plot(phs, 'o', color=color)


def calculate_columns_densities(columns, height, radius):
    '''compute, for each column the cell density, in cells per cubic micron,
    of a cylindrical volume spanning y and centered around it's xz centre '''
    volume = (np.pi * (radius ** 2) * height)
    densities = []

    for c in columns:
        xz = c.positions[:, [0, 2]]
        offset = np.mean(xz, axis=0)
        distance2 = np.sum(np.square(xz - offset), axis=1)
        mask = distance2 < (radius ** 2)
        ncells = np.count_nonzero(mask)
        densities.append(ncells / volume)

    return np.array(densities)


def report_columns_densities(columns_groups, height, radius):
    '''plot column densities for many O1 circuits'''
    plt.figure()
    plt.title('Column density in O1\n(n=%d, core radius=%d microns)' %
              (len(columns_groups), radius))
    plt.xlabel('column id')
    plt.ylabel('density (cells per cubic millimeter)')

    all_densities = []
    for o7 in columns_groups:
        densities = calculate_columns_densities(o7, height, radius) * (10 ** 9)
        all_densities.append(densities)
        plt.plot(densities, ':')

    plt.plot(np.mean(all_densities, axis=0), '-b', label='average', linewidth=2)
    plt.legend(loc='lower right')
