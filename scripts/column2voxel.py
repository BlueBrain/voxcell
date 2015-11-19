#!/usr/bin/env python
'''Attempt to take an circuit.mvd2 release, and transform it into
the voxel representation of densities that can be used by BrainBuilder
'''
import argparse

import numpy as np

from collections import defaultdict
from brainbuilder.utils import genbrain as gb

X, Y, Z = (0, 1, 2)


def load_mvd2_layers_positions(mvd2_path):
    '''load the positions of the neurons from the MVD2 file

    '''
    # TODO: This code is duplicated from mgevaert's mvd2_2_bin.py, remove this when it has a home
    MVD_SECTIONS = [s.lower() for s in ("HEADER", "Neurons Loaded", "MicroBox Data",
                                        "MiniColumnsPosition", "CircuitSeeds",
                                        "MorphTypes", "ElectroTypes", "FOOTER")]
    positions = []
    layers = []
    with open(mvd2_path) as fd:
        while 'Neurons Loaded' not in fd.readline():
            pass

        for line in fd.readlines():
            line = line.strip()
            if line.lower() in MVD_SECTIONS:
                break

            (_, _, _, _, layer, _, _, x, y, z, _, _) = line.split()
            positions.append((float(x), float(y), float(z)))
            layers.append(int(layer))

    positions = np.array(positions)
    layers = np.array(layers)
    return layers, positions


def translate_positions_positive(positions, halo=25.):
    '''Old column was centred at (0,0,0), we want all voxels to be positive, so we shift
       the whole column

       positions(numpy array): positions of soma
       halo(float): distance from edges that column is offset (so it doesn't exactly touch 0
    '''
    min_positions = np.amin(positions, axis=0)
    offset = np.zeros_like(min_positions)

    mask = min_positions < 0.
    offset[mask] = np.abs(min_positions[mask]) + halo
    offset[~mask] = halo - np.abs(min_positions[~mask])

    return positions + offset


def create_annotations(positions, layers, dimensions, spacing, layer_map):
    '''create an annotation map like from AIBS: each voxel is tagged with a specific value

    Args:
        positions(numpy array): positions of soma
        layers(list): layer int for each of the position rows
        dimensions(tuple(x, y, z)) of
        layer_map(dict): translates layer numbers -> annotation value

    Returns:
        numpy array of annotations
    '''
    annotations = np.zeros(dimensions, dtype=np.uint32)
    counts = defaultdict(lambda: [0] * 6)
    voxel_indices = gb.cell_positions_to_voxel_indices(positions, [spacing] * 3)

    for idx, layer in zip(voxel_indices, layers):
        counts[tuple(idx)][layer] += 1

    for idx, layer_count in counts.iteritems():
        layer_value = layer_count.index(max(layer_count))
        annotations[idx] = layer_map[layer_value]

    annotations[annotations <= 0] = layer_map['VOID']

    min_ind = np.amin(voxel_indices, axis=0)
    max_ind = np.amax(voxel_indices, axis=0)
    annotations[min_ind[X]:max_ind[X], max_ind[Y], min_ind[Z]:max_ind[Z]] = layer_map['PIA']

    return annotations


def column2voxel(mvd2_path, column_name, spacing):
    '''
    Args:
        mvd2_path(str path): path to mvd2 file
        column_name(str): name of the column, will be used to output files
    '''
    layers, positions = load_mvd2_layers_positions(mvd2_path)
    positions = translate_positions_positive(positions)

    dimensions = np.ceil(np.amax(positions, axis=0) / spacing).astype(dtype=np.uint32)

    density = gb.cell_density_from_positions(positions, dimensions, [spacing] * 3, dtype=np.float32)
    # density = density / np.sum(density)
    gb.VoxelData(density, [spacing] * 3).save_metaio(column_name + '_density.mhd')

    # from bbp_hierarchy.json
    layer_map = {0: 21,  # Primary somatosensory area, lower limb, layer 1
                 1: 22,  # Primary somatosensory area, lower limb, layer 2
                 2: 23,  # Primary somatosensory area, lower limb, layer 3
                 3: 24,  # Primary somatosensory area, lower limb, layer 4
                 4: 25,  # Primary somatosensory area, lower limb, layer 5
                 5: 26,  # Primary somatosensory area, lower limb, layer 6
                 'PIA': 0,
                 'VOID': 1,
                 }

    annotations = create_annotations(positions, layers, dimensions, spacing, layer_map)
    gb.VoxelData(annotations, [spacing] * 3).save_metaio(column_name + '_annotations.mhd')


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser(description='Convert BBP column to voxel density')

    parser.add_argument('-m', '--mvd', required=True,
                        help='Path to MVD file')
    parser.add_argument('-n', '--name', required=True,
                        help='Name of column')
    parser.add_argument('-s', '--spacing', required=True, type=int,
                        help='Voxel spacing')
    return parser


def main(args):
    '''main function'''
    column2voxel(args.mvd, args.name, args.spacing)


if __name__ == '__main__':
    main(get_parser().parse_args())
