'''main circuit building workflow'''
import argparse
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
from brainbuilder.export_viewer import export_viewer
from brainbuilder.export_mvd2 import export_mvd2


import logging
L = logging.getLogger(__name__)


# pylint: disable=R0914
def main(annotations_path, hierarchy_path, atlas_volume_path,
         recipe_filename, neurondb_filename,
         region_name, total_cell_count, output_path):
    '''Workflow steps replacing BlueBuilder.

    Args:
        annotations_path(str path): path to .mhd file containing annotations
        hierarchy_path(str path): path to .json file containing hierarchy mapping
            related to the annotations_path
        atlas_volume_path(str path): path to .mhd file containing atlas volume density
        region_name(str): region of the brain to build, looked up in hierarchy
        total_cell_count(int): number of cells to place
        output_path(str path): directory where output files will go

    The logic is organised around voxel data.
    The idea being that voxel data allows me to isolate region-specific logic from the general
    circuit building workflow.

    This leaves out Morphology Repair and Neuron model fitting (OptimizerFramework/ModelManagement).

    The variables imply a data dependency: note that many of the steps could be sorted differently.
    '''

    L.debug('Creating brain region: "%s", cell count: %d', region_name, total_cell_count)

    annotation = gb.MetaIO.load(annotations_path)

    hierarchy = gb.load_hierarchy(hierarchy_path)['msg'][0]

    full_density = gb.MetaIO.load(atlas_volume_path)

    rotation_ranges = ((0, 0), (0, 2 * np.pi), (0, 0))

    voxel_dimensions = full_density.mhd['ElementSpacing']

    # transform BBP recipies into voxel data:
    recipe_sdist = bbp.load_recipe_as_spatial_distribution(recipe_filename,
                                                           annotation.raw, hierarchy, region_name)

    synapse_class_sdist = tt.reduce_distribution_collection(recipe_sdist, 'synapse_class')

    neuron_sdist = bbp.load_neurondb_v4_as_spatial_distribution(neurondb_filename, annotation.raw,
                                                                hierarchy, region_name,
                                                                percentile=0.92)

    # main circuit building workflow:

    density_raw = select_region(annotation.raw, full_density.raw, hierarchy, region_name)

    orientation_field = compute_sscx_orientation_fields(annotation, hierarchy, region_name)

    cells = gb.CellCollection()

    cells.positions = cell_positioning(density_raw, voxel_dimensions, total_cell_count)

    cells.orientations = assign_orientations(cells.positions, orientation_field, voxel_dimensions)

    cells.orientations = randomise_orientations(cells.orientations, rotation_ranges)

    chosen_synapse_class = assign_synapse_class_from_spatial_dist(cells.positions,
                                                                  synapse_class_sdist,
                                                                  voxel_dimensions)
    cells.add_properties(chosen_synapse_class)

    chosen_me = assign_metype(cells.positions, cells.properties.synapse_class,
                              recipe_sdist, voxel_dimensions)
    cells.add_properties(chosen_me)

    chosen_morphology = assign_morphology(cells.positions, cells.properties[['mtype', 'etype']],
                                          neuron_sdist, voxel_dimensions)
    cells.add_properties(chosen_morphology)

    acronym = gb.find_in_hierarchy(hierarchy, 'name', region_name)[0]['acronym']
    export_viewer(joinp(output_path, 'intermediates_%s_%d' % (acronym, total_cell_count)),
                  voxel_dimensions, orientation_field, cells)

    # export data to file formats from the BBP pipeline:
    circuit_path = export_mvd2(output_path, 'mpath', cells)

    cells.save(joinp(output_path, 'cells.h5'))

    return circuit_path


def get_region_names(hierarchy):
    '''retuns the names of all the regions'''
    names = sorted(gb.get_in_hierarchy(hierarchy, 'name'))
    return names


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser(description='Create a brain')

    parser.add_argument('-a', '--annotations', required=True,
                        help='path to annotations MHD')
    parser.add_argument('-i', '--hierarchy', required=True,
                        help='path to hierarchy json')
    parser.add_argument('-d', '--density', required=True,
                        help='path to density MHD')
    parser.add_argument('-p', '--recipe', required=True,
                        help='BBP Recipe .xml')
    parser.add_argument('-n', '--neurondb', required=True,
                        help='BBP Neuron DB')
    parser.add_argument('-r', '--region', default='Primary somatosensory area, lower limb',
                        help='Name of region to use')
    parser.add_argument('-c', '--cellcount', type=int, required=True,
                        help='Number of cells to place')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for BBP file formats')
    parser.add_argument('-l', '--lsregion', default=False, action='store_true',
                        help='List all know regions')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])
    if args.lsregion:
        print '\n'.join(get_region_names(args.hierarchy))
        sys.exit(0)

    main(args.annotations, args.hierarchy, args.density,
         args.recipe, args.neurondb,
         args.region, args.cellcount, args.output)
