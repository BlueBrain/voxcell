'''compatibility functions with existing BBP formats'''
import numpy as np
from collections import OrderedDict, defaultdict
from lxml import etree
from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import traits as tt

import logging
L = logging.getLogger(__name__)


def map_regions_to_layers(hierarchy, region_name):
    '''map regions in the hierarchy to the layer divisions used in BBP
    returns a dictionary where the key is the region id according to the Allen Brain data
    and the value is a tuple of the integer indices of the 6 layers used in BBP: 1, 2, 3, 4, 5, 6
    '''

    sub_area_names = gb.collect_in_hierarchy(hierarchy, 'name', region_name, 'name')

    layer_mapping = {
        'layer 1': (1,),
        'layer 2/3': (2, 3),
        'layer 2': (2, ),
        'layer 3': (3, ),
        'layer 4/5': (4, 5),
        'layer 4': (4,),
        'layer 5/6': (5, 6),
        'layer 5': (5,),
        'layer 6': (6,),
        'layer 6a': (6,),
        'layer 6b': (6,),
        ', 6a': (6,),
        ', 6b': (6,),
    }
    layer_groups = {}
    for subarea in sub_area_names:
        for name, indices in layer_mapping.items():
            if subarea.lower().endswith(name):
                area = gb.find_in_hierarchy(hierarchy, 'name', subarea)
                layer_groups[area[0]['id']] = indices

    return layer_groups


def load_recipe_as_layer_distributions(recipe_filename):
    '''take a BBP builder recipe and return the probability distributions for each type

    returns a dictionary where the keys are layer ids (an integer: 1, 2, 3, 4, 5, 6)
    and the value is a list of tuples of the form (probability, type_def) where
    probability is a float in [0, 1]
    type_def is dictionary with the properties: mtype, etype, mClass, sClass
    '''
    recipe_tree = etree.parse(recipe_filename)

    sclass_alias = {
        'INH': 'inhibitory',
        'EXC': 'excitatory'
    }

    layer_distributions = {}
    for layer in recipe_tree.findall('NeuronTypes')[0].getchildren():

        for structural_type in layer.getchildren():
            if structural_type.tag == 'StructuralType':

                for electro_type in structural_type.getchildren():
                    if electro_type.tag == 'ElectroType':

                        percentage = (float(structural_type.attrib['percentage']) / 100 *
                                      float(electro_type.attrib['percentage']) / 100)

                        layer_id = int(layer.attrib['id'])
                        type_def = {
                            'mtype': structural_type.attrib['id'],
                            'etype': electro_type.attrib['id'],
                            'mClass': structural_type.attrib['mClass'],
                            'sClass': sclass_alias[structural_type.attrib['sClass']]
                        }
                        layer_distributions.setdefault(layer_id, []).append((percentage, type_def))

    return layer_distributions


def combine_distributions(distributions):
    '''take a list of distributions and return a single one normalised

    a distribution is represented as a dictionary (type_id, probability) where
    type_id is a hashable type (example: integer) representing a type definition
    probability is a float in [0, 1]
    '''
    combined = defaultdict(float)
    for distribution in distributions:
        for type_id, probability in distribution.items():
            combined[type_id] += probability / float(len(distributions))

    total = sum(combined.values())
    return dict((type_id, probability / total) for type_id, probability in combined.iteritems())


def transform_into_spatial_distribution(annotation_raw, layer_distributions, region_layers_map):
    '''take distributions grouped by layer ids and a map from regions to layers
    and build a volumetric dataset that contains the same distributions

    returns a SpatialDistribution object where the properties of the traits_collection are:
    mtype, etype, mClass, sClass
    '''

    traits_field = np.ones_like(annotation_raw) * -1

    unique_distributions = OrderedDict()
    unique_type_defs = OrderedDict()

    for region_id, layer_ids in region_layers_map.items():

        distributions = []
        for layer_id in layer_ids:
            dist = {}
            for p, t in layer_distributions[layer_id]:
                hashable_type_def = tuple(t.items())
                if hashable_type_def not in unique_type_defs:
                    unique_type_defs[hashable_type_def] = len(unique_type_defs)

                dist[unique_type_defs[hashable_type_def]] = p

            distributions.append(dist)

        hashable_dist = tuple(combine_distributions(distributions).items())

        if hashable_dist not in unique_distributions:
            unique_distributions[hashable_dist] = len(unique_distributions)

        traits_field[annotation_raw == region_id] = unique_distributions[hashable_dist]

    return tt.SpatialDistribution(traits_field,
                                  [dict(dist) for dist in unique_distributions.keys()],
                                  [dict(dist) for dist in unique_type_defs.keys()])


def load_recipe_as_spatial_distributions(recipe_filename, annotation_raw, hierarchy, region_name):
    '''load the bbp recipe and return a spatial voxel-based distribution
    returns: see transform_into_spatial_distribution
    '''
    region_layers_map = map_regions_to_layers(hierarchy, region_name)

    layer_distributions = load_recipe_as_layer_distributions(recipe_filename)

    return transform_into_spatial_distribution(annotation_raw,
                                               layer_distributions,
                                               region_layers_map)
