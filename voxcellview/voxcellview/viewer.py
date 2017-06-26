'''functions to export the results of brain building to an external viewer'''

import colorsys
import logging

import numpy as np

L = logging.getLogger(__name__)
RGB = slice(3)


def get_cell_color(cells, attribute, input_color_map=None):
    '''compute an array with colors for each cell depending on a given attribute

    Args:
        cells(indexable): cells
        attribute(str): attribute of cell collection, 'position' is a special attribute
        which shows a point cloud
        input_color_map(callable) that returns a list of 3 elements corresponding
        to RGB value between 0 and 1, when called with 'position', an array of x/y/z
        is passed, otherwise the value of a attribute row is passed

    Returns:
        np.array of RGBA values between 0 and 1
    '''
    if attribute == 'position':
        if input_color_map is None:
            def constant_white(_):
                '''always returns white'''
                return [1, 1, 1]
            input_color_map = constant_white
        values = cells[['x', 'y', 'z']].values
    else:
        values = cells[attribute]

    ret = np.ones((len(cells), 4), dtype=np.float32)
    if input_color_map is not None:
        assert callable(input_color_map), 'input_color_map must be a callable'
        for i, value in enumerate(values):
            ret[i, RGB] = input_color_map(value)
    else:
        unique_values, indices = np.unique(values, return_inverse=True)
        colors = np.array([colorsys.hsv_to_rgb(i, 1, 1)
                           for i in np.linspace(0, 1, len(unique_values), endpoint=False)])
        ret[:, RGB] = colors[indices, :]

    return ret


def serialize_points(cells, attribute, color_map):
    '''convert a collection of cells to binary to a format that can be loaded by the JS viewer
        The serialized data format is a long array of float32 representing,
        for each cell, its position and color:
        [x_0, y_0, z_0, r_0, g_0, b_0,
         x_1, y_1, z_1, r_1, g_1, b_1,
         etc]
        Color components are in the range [0, 1]
    '''
    L.debug("serializing %d points", len(cells))
    colors = get_cell_color(cells, attribute, color_map)
    return np.hstack([
        cells[['x', 'y', 'z']],
        colors
    ]).astype(np.float32)


def export_points(filename, cells, attribute, color_map=None):
    '''save a collection of cells to binary to a format that can be loaded by the JS viewer'''
    block = serialize_points(cells, attribute, color_map)
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

    positions = np.array(idx).transpose()  # pylint: disable=no-member
    positions *= voxel_dimensions
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


def export_positions_vectors(filename, cells, attribute, color_map=None):
    ''' export position along with vectors to a binary file of float 32'''
    block = serialize_positions_vectors(cells, attribute, color_map)
    block.tofile(filename)


def serialize_positions_vectors(cells, attribute, color_map=None):
    ''' serialize a position, orientation, color for all cells '''
    colors = get_cell_color(cells, attribute, color_map)
    return np.hstack([
        cells[['x', 'y', 'z']],
        np.vstack([m.flatten() for m in cells['orientation']]),
        colors
    ]).astype(np.float32)


def export_strings(filename, all_strings):
    ''' export an array of strings to a txt file '''
    np.savetxt(filename, all_strings, delimiter=" ", fmt="%s")
