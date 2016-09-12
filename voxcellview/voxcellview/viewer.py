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


def get_cell_color(cells, attribute, input_color_map):
    '''compute an array with colors for each cell depending on a given attribute.
    input_color_map is a callable that returns a list of 3 elements corresponding
    to r,g,b value between 0 and 1.
    '''
    def add_default_opacity(array):
        if len(array) == 3:
            array = np.append(array, 1.0)
        return array

    if input_color_map:
        ret = [add_default_opacity(input_color_map(t)) for t in cells.properties[attribute]]
        return ret

    colormap = load_traits_colormap(os.path.join(DATA_FOLDER, 'colormap.json'))
    colormap = colormap[attribute]

    def normalize_rgb(array):
        array = array.astype(float)
        array[:3] /= 255.0
        ret =  add_default_opacity(array)
        return ret

    if isinstance(colormap, dict):
        return [normalize_rgb(np.array(colormap[t])) for t in cells.properties[attribute]]
    else:
        # color doesn't depend on the value of the attribute
        constant_color = normalize_rgb(np.array(colormap))
        return [constant_color] * len(cells.positions)


def serialize_points(cells, attribute, color_map):
    '''convert a collection of cells to binary to a format that can be loaded by the JS viewer
        The serialized data format is a long array of float32 representing,
        for each cell, its position and color:
        [x_0, y_0, z_0, r_0, g_0, b_0,
         x_1, y_1, z_1, r_1, g_1, b_1,
         etc]
        Color components are in the range [0, 1]
    '''
    L.debug("serializing %d points", cells.positions.shape[0])
    colors = get_cell_color(cells, attribute, color_map)
    return np.append(cells.positions, colors, axis=1).astype(np.float32)


def export_points(filename, cells, attribute):
    '''save a collection of cells to binary to a format that can be loaded by the JS viewer'''
    block = serialize_points(cells, attribute)
    block.tofile(filename)


def serialize_vectors(positions, vectors):
    '''convert a bunch of vectors to binary to a format that can be loaded by the JS viewer
      The serialized data format is: a long array of float32 representing,
      for each vector, its position, initial color, vector and end color:
        [x_0, y_0, z_0, r0_0, g0_0, b0_0, i_0, j_0, k_0, r1_0, g1_0, b1_0,
         x_1, y_1, z_1, r0_1, g0_1, b0_1, i_1, j_1, k_1, r1_1, g1_1, b1_1,
         etc]
      Color components are in the range [0, 1]
    '''
    L.debug("serializing %d vectors", positions.shape[0])
    colors1 = np.abs(vectors)
    colors0 = colors1 * 0.5
    p0_block = np.append(positions, colors0, axis=1).astype(np.float32)
    p1_block = np.append(vectors, colors1, axis=1).astype(np.float32)
    return np.append(p0_block, p1_block, axis=1).astype(np.float32)


def export_vectors(filename, positions, vectors):
    '''save a bunch of vectors to binary to a format that can be loaded by the JS viewer.
    The color of each vector encodes the XYZ components'''
    block = serialize_vectors(positions, vectors)
    block.tofile(filename)


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
    block = serialize_vector_field(field, point_count, voxel_dimensions)
    block.tofile(filename)


def serialize_vector_field(field, point_count, voxel_dimensions):
    ''' convert a vector field to a format that can be loaded by the JS viewer'''
    positions, vectors = sample_vector_field(field, point_count, voxel_dimensions)
    block = serialize_vectors(positions, vectors)
    return block


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
