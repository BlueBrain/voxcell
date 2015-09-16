'''export results to MVD2 file
'''
import bisect
import logging
import os

from collections import namedtuple
from itertools import izip
from os.path import join as joinp

from brainbuilder.version import VERSION

L = logging.getLogger(__name__)
X, Y, Z = (0, 1, 2)
MAP_EXC_INH = {'excitatory': 'EXC',
               'inhibitory': 'INH',
               }


#MVD2 Spec
#https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/Circuit+-+circuit.mvd2

#MorphType as per MVD2 spec
MorphType = namedtuple('MorphType', 'MName type sclass')

#Neuron information encoded per line as per MVD2 spec
MVD2Neuron = namedtuple('MVD2Neuron', 'name database hyperColumn miniColumn layer '
                        'morphology electrophysiology x y z rotation metype')

# From
# /gpfs/bbp.cscs.ch/project/proj1/entities/bionames/
# SomatosensoryCxS1-v5.r0/builderRecipeAllPathways.xml
LAYER_THICKNESS = [700.37845971,  # 6
                   525.05585701,  # 5
                   189.57183895,  # 4
                   352.92508322,  # 3
                   148.87602025,  # 2
                   164.94915873,  # 1
                   ]
LAYER_BOUNDARIES = [sum(LAYER_THICKNESS[:i + 1])
                    for i in range(len(LAYER_THICKNESS) - 1)]


def _position_to_layer(y):
    '''find the layer, based on the y coordinate
       returns a layer from 1 - 6
    '''
    return 6 - bisect.bisect(LAYER_BOUNDARIES, y)


def _create_MVD2Neuron(morph_types, electro_types, name=None, # pylint: disable=R0913
                       morphology=None, electrophysiology=None,
                       x=0.0, y=0.0, z=0.0, rotation=0.0, layer=None,
                       metype=None, database=None, hyperColumn=None, miniColumn=None):
    '''Simplify creation fo MVD2Neuron line

    Note: Layer is stored as stored 0 based
    '''
    database = database or 0
    hyperColumn = hyperColumn or 0
    miniColumn = miniColumn or 0
    #guess based on morphology
    layer = _position_to_layer(y) - 1 if layer is None else layer
    metype = metype or '%s_%s_%s' % (electrophysiology, morphology, name)

    return MVD2Neuron(name=name, database=database, hyperColumn=hyperColumn, miniColumn=miniColumn,
                      layer=layer, morphology=morph_types[morphology],
                      electrophysiology=electro_types[electrophysiology],
                      rotation=rotation, x=x, y=y, z=z, metype=metype)


def mvd2_string(self):
    '''encode the neuron string as per MVD2 spec'''
    return ('{name} {database} {hyperColumn} {miniColumn} {layer} {morphology} '
            '{electrophysiology} {x} {y} {z} {rotation} {metype}').format(
                **self._asdict()) # pylint: disable=W0212
MVD2Neuron.mvd2_string = mvd2_string


def _create_mvd2_header(morphology_path):
    '''retunrs mvd2 header'''
    return '''\
Application:'BrainBuilder {version}'
{morphology_path}
/unknown/
'''.format(version=VERSION,
           morphology_path=morphology_path)


def _get_mvd2_neurons(morph_types_by_name, electro_types, chosen_morphology, positions,
                      chosen_mtype, chosen_etype):
    '''return all the MVD2Neuron objects used in circuit'''
    #TODO: Set the proper y-rotation for cell, currently set to 0.0
    return [_create_MVD2Neuron(morph_types_by_name, electro_types, name=morph,
                               morphology=mtype, electrophysiology=etype,
                               rotation=0.0, x=pos[X], y=pos[Y], z=pos[Z])
            for (morph, pos, mtype, etype) in izip(chosen_morphology,
                                                   positions,
                                                   chosen_mtype,
                                                   chosen_etype)]


def _create_mvd2(mvd2_path, morphology_path, positions,
                 chosen_synapse_class, chosen_mtype, chosen_etype, chosen_morphology):
    '''create an MVD2 file'''
    #mapping from name to position in list
    #TODO need to know if cells are PYR/INT, set all to PYR at the moment
    morph_types = set(MorphType(m, 'PYR', MAP_EXC_INH[sc])
                      for m, sc in izip(chosen_mtype, chosen_synapse_class))
    morph_types = dict((morph_type, i) for i, morph_type in enumerate(morph_types))
    morph_types_by_name = dict((morph_type.MName, i) for morph_type, i in morph_types.items())

    #mapping from name to position in list
    electro_types = set(chosen_etype)
    electro_types = dict((electro_type, i) for i, electro_type in enumerate(electro_types))

    neurons = _get_mvd2_neurons(morph_types_by_name, electro_types,
                                chosen_morphology, positions, chosen_mtype, chosen_etype)

    with open(mvd2_path, 'w') as fd:
        fd.write(_create_mvd2_header(morphology_path))

        fd.write('MorphTypes\n')
        for _, k in sorted((v, k) for k, v in morph_types.items()):
            fd.write('{MName} {type} {sclass}\n'.format(**k._asdict())) # pylint: disable=W0212

        fd.write('ElectroTypes\n')
        for _, k in sorted((v, k) for k, v in electro_types.items()):
            fd.write(k + '\n')

        fd.write('Neurons Loaded\n')
        for n in neurons:
            fd.write(n.mvd2_string() + '\n')

        #TODO: Is this needed?
        #fd.write('MiniColumnsPosition\n')
        #for mt in morph_types:
        #    fd.write(mt)

        #TODO, do we need these?
        #fd.write('CircuitSeeds\n')
        #fd.write('0.0 1.0 2.0\n')


def export_mvd2(directory, morphology_path, cells):
    '''Rest of mesobuilder steps to make the circuit readable by bluepy:
    create a CircutConfig, create sqlite, etc.

    Args:
        directory(str path): location where circuit is created
        morphology_path(str path): directory of h5 morphologies
        cells: CellCollection object with at least: positions, sClass, mtype, etype, morphology

    Returns:
        MVD2 file
    '''
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError:
            L.exception('Need a directory to put circuit')

    mvd2_path = joinp(directory, 'circuit.mvd2')

    _create_mvd2(mvd2_path, morphology_path,
                 cells.positions,
                 cells.properties.sClass,
                 cells.properties.mtype,
                 cells.properties.etype,
                 cells.properties.morphology)
    return mvd2_path
