'''functions to export the results of brain building to an external viewer'''

import os
import json
import numpy as np
from collections import defaultdict


DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')


def load_traits_colormap(filepath):
    '''a map of colors grouped by attributes (mtype,etype,sclass,etc) connecting every
    value of the attribute to a color'''
    with open(filepath) as f:
        colormap = json.load(f)

    # TODO pick up the "default" key if defined in the colormap.json
    return defaultdict(lambda: defaultdict(lambda: np.random.randint(256, size=3)),
                       colormap)


def serialize_points(filename, positions, colors):
    '''save a bunch of points to binary to a format that can be loaded by the JS viewer'''
    block = np.append(positions, colors, axis=1).astype(np.float32)
    block.tofile(filename)


def export_points(filename, positions, chosen_traits, attribute):
    '''save a bunch of points to binary to a format that can be loaded by the JS viewer'''
    colormap = load_traits_colormap(os.path.join(DATA_FOLDER, 'colormap.json'))
    colors = [np.array(colormap[attribute][t]) / 255.0 for t in chosen_traits]
    serialize_points(filename, positions, colors)
