'''functions to export the results of brain building to an external viewer'''

import os
import json
import numpy as np
from collections import defaultdict

import logging
L = logging.getLogger(__name__)


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


def load_traits_colormap(filepath):
    '''a map of colors grouped by attributes (mtype,etype,sclass,etc) connecting every
    value of the attribute to a color'''
    with open(filepath) as f:
        colormap = json.load(f)

    # TODO pick up the "default" key if defined in the colormap.json
    return defaultdict(lambda: defaultdict(lambda: np.random.randint(256, size=3)),
                       colormap)


def get_cell_color(cells, attribute):
    '''compute an array with colors for each cell depending on a given attribute'''
    colormap = load_traits_colormap(os.path.join(DATA_FOLDER, 'colormap.json'))
    return [np.array(colormap[attribute][t]) / 255.0 for t in cells.properties[attribute]]


def serialize_points(filename, positions, colors):
    '''save a bunch of points to binary to a format that can be loaded by the JS viewer'''
    L.debug("saving %d points in %s", positions.shape[0], filename)
    block = np.append(positions, colors, axis=1).astype(np.float32)
    block.tofile(filename)


def export_points(filename, cells, attribute):
    '''save a bunch of points to binary to a format that can be loaded by the JS viewer'''
    colors = get_cell_color(cells, attribute)
    serialize_points(filename, cells.positions, colors)


def serialize_vectors(filename, positions, vectors, colors0, colors1):
    '''save a bunch of vectors to binary to a format that can be loaded by the JS viewer'''
    L.debug("saving %d vectors in %s", positions.shape[0], filename)
    p0_block = np.append(positions, colors0, axis=1).astype(np.float32)
    p1_block = np.append(vectors, colors1, axis=1).astype(np.float32)
    block = np.append(p0_block, p1_block, axis=1).astype(np.float32)
    block.tofile(filename)


def export_vectors(filename, positions, vectors):
    '''save a bunch of vectors to binary to a format that can be loaded by the JS viewer.
    The color of each vector encodes the XYZ components'''
    colors1 = np.abs(vectors)
    colors0 = colors1 * 0.5
    serialize_vectors(filename, positions, vectors, colors0, colors1)


def sample_vector_field(field, point_count, voxel_dimensions):
    '''generate a list of points and vectors from a vector field'''
    mask = np.any(field != 0, axis=-1)

    valid_total_count = np.count_nonzero(mask)
    point_count = min(point_count, valid_total_count)
    chosen = np.random.choice(np.arange(valid_total_count), size=point_count, replace=False)
    L.debug("exporting %d of %d possible points", point_count, valid_total_count)

    idx = np.nonzero(mask)
    idx = tuple(component[chosen] for component in idx)

    vectors = field[idx]
    vectors = vectors / np.sqrt(np.square(vectors).sum(axis=1))[:, np.newaxis]

    positions = np.array(idx).transpose() * voxel_dimensions
    return positions, vectors


def export_vector_field(filename, field, point_count, voxel_dimensions):
    '''save a vector field to binary to a format that can be loaded by the JS viewer'''
    positions, all_vectors = sample_vector_field(field, point_count, voxel_dimensions)
    export_vectors(filename, positions, all_vectors)


def export_positions_vectors(filename, cells, attribute):
    ''' export position along with vectors to a binary file of float 32'''
    vectors = cells.orientations.reshape((cells.orientations.shape[0],
                                          np.prod(cells.orientations.shape[1:])))

    colors = get_cell_color(cells, attribute)

    reduced_all = reduce(lambda v0, v1: np.append(v0, v1, axis=-1),
                         (cells.positions, vectors, colors))

    reduced_all.astype(np.float32).tofile(filename)


def export_strings(filename, all_strings):
    ''' export an array of strings to a txt file '''
    np.savetxt(filename, all_strings, delimiter=" ", fmt="%s")
