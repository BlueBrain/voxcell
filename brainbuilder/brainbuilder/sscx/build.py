#!/usr/bin/env python

""" Build SSCX column based on BBP recipe. """

import os
import argparse
import logging

import numpy as np
import pandas as pd

from voxcell.core import CellCollection
from voxcell.positions import create_cell_positions

from brainbuilder.utils import bbp

from brainbuilder.sscx.annotation import build_column_atlas


L = logging.getLogger(__name__)


def build_cell_collection(cell_count, annotation, density, recipe_sdist, neuron_sdist):
    """ Build CellCollection based on annotation, density and properties spatial distributions. """

    cells = CellCollection()

    L.debug("Assigning %d cell positions...", cell_count)
    cells.positions = create_cell_positions(density, cell_count)

    L.debug("Assigning layers...")
    cells.add_properties(
        pd.DataFrame({'layer': annotation.lookup(cells.positions).astype(np.str)})
    )

    L.debug("Assigning mtype / etype / synapse_class / morph_class...")
    cells.add_properties(
        recipe_sdist.collect(
            positions=cells.positions,
            preassigned=cells.properties[['layer']],
            names=['mtype', 'etype', 'synapse_class', 'morph_class']
        )
    )

    L.debug("Assigning morpologies...")
    cells.add_properties(
        neuron_sdist.collect(
            positions=cells.positions,
            preassigned=cells.properties[['layer', 'mtype', 'etype']],
            names=['morphology']
        )
    )

    # orientation
    # TODO: ???

    return cells


def main(recipe_filename, neurondb_filename, output_dir):
    """ Build and save annotation and CellCollection based on recipe and neurondb. """

    L.debug("Building column atlas...")
    annotation, _ = build_column_atlas(recipe_filename)

    annotation.save_nrrd(os.path.join(output_dir, "column.nrrd"))

    region_map = {x: str(x,) for x in range(1, 7)}

    density = bbp.load_recipe_density(recipe_filename, annotation, region_map)

    cell_count = bbp.get_total_neurons(recipe_filename)

    recipe_sdist = bbp.load_recipe_as_spatial_distribution(
        recipe_filename, annotation, region_map
    )

    L.debug("Loading neurondb...")
    neuron_sdist = bbp.load_neurondb_v4_as_spatial_distribution(
        neurondb_filename, annotation, region_map, percentile=0.92
    )

    cells = build_cell_collection(cell_count, annotation, density, recipe_sdist, neuron_sdist)
    cells.save(os.path.join(output_dir, "cells.mvd3"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a brain')
    parser.add_argument(
        '-p', '--recipe',
        required=True,
        help='BBP Recipe .xml'
    )
    parser.add_argument(
        '-n', '--neurondb',
        required=True,
        help='BBP Neuron DB'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for BBP file formats'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='-v for INFO, -vv for DEBUG'
    )
    args = parser.parse_args()

    loglevel = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=loglevel)

    main(args.recipe, args.neurondb, args.output)
