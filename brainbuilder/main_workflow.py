'''main circuit building workflow'''
import argparse
import os
import sys

import numpy as np

from os.path import join as joinp

from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import bbp
from brainbuilder.utils import traits as tt
from brainbuilder.orientation_fields import compute_sscx_orientation_fields
from brainbuilder.select_region import select_region
from brainbuilder.cell_positioning import cell_positioning
from brainbuilder.assignment_synapse_class import assign_synapse_class_from_spatial_dist
from brainbuilder.assignment_metype import assign_metype
from brainbuilder.assignment_morphology import assign_morphology
from brainbuilder.assignment_orientation import assign_orientations
from brainbuilder.assignment_orientation import randomise_orientations
from brainbuilder.export_bbp import export_for_bbp

import logging
L = logging.getLogger(__name__)


def main(data_dir, region_name, total_cell_count):  # pylint: disable=R0914
    '''
    Most of the workflow steps here replace BlueBuilder.

    The logic is organised around voxel data.
    The idea being that voxel data allows me to isolate region-specific logic from the general
    circuit building workflow.

    This leaves out Morphology Repair and Neuron model fitting (OptimizerFramework/ModelManagement).

    The variables imply a data dependency: note that many of the steps could be sorted differently.
    '''

    L.debug('Creating brain, data_dir: "%s", region: "%s", cell count: %d',
            data_dir, region_name, total_cell_count)
    # workflow arguments (need to be provided by the user)

    annotation = gb.MetaIO.load(joinp(data_dir, 'P56_Mouse_annotation/annotation.mhd'),
                                joinp(data_dir, 'P56_Mouse_annotation/annotation.raw'))

    hierarchy = gb.load_hierarchy(
        os.path.join(data_dir, 'P56_Mouse_annotation/annotation_hierarchy.json'))['msg'][0]

    full_density = gb.MetaIO.load(joinp(data_dir, 'atlasVolume/atlasVolume.mhd'),
                                  joinp(data_dir, 'atlasVolume/atlasVolume.raw'))

    recipe_filename = os.path.join(data_dir, 'bbp_recipe/builderRecipeAllPathways.xml')
    neurondb_filename = os.path.join(data_dir, 'prod_NeuronDB_19726.dat')

    #total_cell_count = 4000000
    #rotation_ranges = ((0, 0), (0, 2 * np.pi), (0, 0))
    #region_acronym = 'SSp-ll'
    rotation_ranges = ((0, 0), (0, 2 * np.pi), (0, 0))
    #inhibitory_fraction = 0.10

    voxel_dimensions = full_density.mhd['ElementSpacing']

    # transform BBP recipies into voxel data:

    recipe_sdist = bbp.load_recipe_as_spatial_distribution(recipe_filename,
                                                           annotation.raw, hierarchy, region_name)

    sclass_sdist = tt.reduce_distribution_collection(recipe_sdist, 'sClass')

    neuron_sdist = bbp.load_neurondb_v4_as_spatial_distribution(neurondb_filename, annotation.raw,
                                                                hierarchy, region_name)

    # main circuit building workflow:

    density_raw = select_region(annotation.raw, full_density.raw, hierarchy, region_name)

    orientation_field = compute_sscx_orientation_fields(annotation, hierarchy, region_name)

    positions = cell_positioning(density_raw, voxel_dimensions, total_cell_count)

    orientations = assign_orientations(positions, orientation_field, voxel_dimensions)

    orientations = randomise_orientations(orientations, rotation_ranges)

    chosen_synapse_class = assign_synapse_class_from_spatial_dist(positions, sclass_sdist,
                                                                  voxel_dimensions)

    chosen_me = assign_metype(positions, chosen_synapse_class, recipe_sdist, voxel_dimensions)

    chosen_morphology = assign_morphology(positions, chosen_me, neuron_sdist, voxel_dimensions)

    # export data to file formats from the BBP pipeline:

    circuit = export_for_bbp(positions, orientations,
                             (chosen_synapse_class, chosen_me, chosen_morphology))

    return circuit


def get_region_names(data_dir):
    '''retuns the names of all the regions'''
    hierarchy = gb.load_hierarchy(
        joinp(data_dir, 'P56_Mouse_annotation/annotation_hierarchy.json'))['msg'][0]
    names = sorted(gb.get_in_hierarchy(hierarchy, 'name'))
    return names


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser(description='Create a brain')

    parser.add_argument('-d', '--data', default='data', required=True,
                        help='Base path to data directory')
    parser.add_argument('-c', '--cellcount', default=400000, type=int,
                        help='Number of cells to place')
    parser.add_argument('-l', '--lsregion', default=False, action='store_true',
                        help='List all know regions')
    parser.add_argument('-r', '--region', default='Primary somatosensory area, lower limb',
                        help='Name of region to use')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])
    if args.lsregion:
        print '\n'.join(get_region_names(args.data))
        sys.exit(0)

    main(args.data, args.region, args.cellcount)
