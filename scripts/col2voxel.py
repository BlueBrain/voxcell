#!/usr/bin/env python
'''Attempt to take an circuit.mvd2 release, and transform it into
the voxel representation of densities that can be used by BrainBuilder
'''
import argparse

import numpy as np

from brainbuilder.utils import genbrain as gb


def load_mvd2_positions(mvd2_path):
    '''load the positions of the neurons from the MVD2 file

    '''
    #TODO: This code is duplicated from mgevaert's mvd2_2_bin.py, remove this when it has a home
    MVD_SECTIONS = [s.lower() for s in ("HEADER", "Neurons Loaded", "MicroBox Data",
                                        "MiniColumnsPosition", "CircuitSeeds",
                                        "MorphTypes", "ElectroTypes", "FOOTER")]
    ret = []
    with open(mvd2_path) as fd:
        while 'Neurons Loaded' not in fd.readline():
            pass

        for line in fd.readlines():
            line = line.strip()
            if line.lower() in MVD_SECTIONS:
                break

            (_, _, _, _, _, _, _, x, y, z, _, _) = line.split()
            ret.append((float(x), float(y), float(z)))
    ret = np.array(ret)
    return ret


def column2voxel(mvd2_path, column_name, dimensions, spacing):
    '''
    Args:
        mvd2_path(str path): path to mvd2 file
        column_name(str): name of the column, will be used to output files
        dimensions(tuple): tuple(x, y, z)
    '''
    positions = load_mvd2_positions(mvd2_path)
    density = gb.cell_density_from_positions(positions, dimensions, dimensions, dtype=np.float32)
    density = density / np.sum(density)

    mhd_filename = column_name + '.mhd'
    raw_filename = column_name + '.raw'
    mhd = gb.get_mhd_info(dimensions, density.dtype.type, spacing, raw_filename)

    gb.save_meta_io(mhd_filename, mhd, raw_filename, density)


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser(description='Convert BBP column to voxel density')

    parser.add_argument('-m', '--mvd', required=True,
                        help='Path to MVD file')
    parser.add_argument('-n', '--name', required=True,
                        help='Name of column')
    return parser


def main(args):
    '''main function'''
    #hardcode this for now
    dimensions = (528, 320, 456)
    spacing = (25, 25, 25)

    column2voxel(args.mvd, args.name, dimensions, spacing)


if __name__ == '__main__':
    main(get_parser().parse_args())
