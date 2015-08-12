'''compatibility functions with existing BBP formats'''
import numpy as np
import itertools
from collections import OrderedDict, defaultdict
from lxml import etree
from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import traits as tt
from scipy.ndimage import distance_transform_edt  # pylint: disable=E0611

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


def transform_recipe_into_spatial_distribution(annotation_raw,
                                               layer_distributions, region_layers_map):
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


def load_recipe_as_spatial_distribution(recipe_filename, annotation_raw, hierarchy, region_name):
    '''load the bbp recipe and return a spatial voxel-based distribution
    returns: see transform_into_spatial_distribution
    '''
    region_layers_map = map_regions_to_layers(hierarchy, region_name)

    layer_distributions = load_recipe_as_layer_distributions(recipe_filename)

    return transform_recipe_into_spatial_distribution(annotation_raw,
                                                      layer_distributions,
                                                      region_layers_map)


def load_neurondb_v4(neurondb_filename):
    '''load a neurondb v4 file as a list of dictionaries where the keys are:
    morphology, layer, mtype, etype, metype, placement_hints
    '''

    morphologies = []

    with open(neurondb_filename) as f:
        for line in f.readlines():
            fields = line.split()
            morphology, layer, mtype, etype, metype = fields[:5]
            placement_hints = fields[5:]
            morphologies.append({'morphology': morphology,
                                 'layer': int(layer),
                                 'mtype': mtype,
                                 'etype': etype,
                                 'metype': metype,
                                 'placement_hints': tuple(placement_hints)})

    return morphologies


def get_morphologies_by_layer(neurondb):
    '''group morphologies by layer

    Args:
        neurondb: a list of dictionaries representing morphologies and their properties.
            (see load_neurondb_4). The only required property is 'layer'

    Returns:
        a dictionary where the keys are layer ids and the values lists of morphologies
    '''
    return dict((l, list(ns)) for l, ns in itertools.groupby(neurondb, lambda m: m['layer']))


def get_morphologies_by_layer_group(morphs_by_layer, layer_ids):
    '''group morphologies by layer group of layers

    Args:
        morphs_by_layer: dictionary where the keys are layer ids and the values are
            lists of morphologies
        layer_ids: a collection of layer ids

    Returns:
        A list of all of the available morphologies for a group of layers
    '''
    return list(itertools.chain(*(morphs_by_layer[layer_id] for layer_id in layer_ids)))


def get_placement_hints_table(morphs):
    '''collect the placement hint scores for a group of morphologies.

    Placement hints are a series of numbers associated with each morphology. This numbers
    represent how good a fit a morphology is to each subsection of space after this has been
    evenly splitted.

    For example, having a morphology with scores [1, 2, 1] means that it is more likely to
    find this morphology in the second third of a space than it is to find it in the first or
    the last thirds.

    The original concept of "space" was layers and they were divided in the Y direction
    (towards pia). This allowed, for example, having morphologies appear only in the bottom
    half of a layer. Now that we are dealing with complex volumes, bottom and top don't mean
    much. Here "space" is just a collection of voxels which can be grouped according to some
    metric (distance to exterior).

    Args:
        morphs: a collection of morphologies.

    Returns:
        A 2D numpy array that contains the placement hint scores for the given morphologies.
        This table has one row for each morphology and one column for each region subdivision
    '''
    subdivision_count = gb.lcmm(len(m['placement_hints']) for m in morphs)

    region_dist_table = np.empty(shape=(len(morphs), subdivision_count))

    for i, m in enumerate(morphs):
        for hint_idx, hint in enumerate(m['placement_hints']):
            repetitions = subdivision_count // len(m['placement_hints'])
            for j in range(repetitions):
                region_dist_table[i, hint_idx * repetitions + j] = hint

    return region_dist_table


def get_region_distributions_from_placement_hints(neurondb, region_layers_map):
    '''for every region, return the list of probability distributions for each potential
    morphology. The probabilites are taken from the placement hint scores.
    There is one distribution for each subdivision of the region and they are sorted
    the same way as the placement hint scores are: from closest to pia to furthest to pia

    Returns:
        A dictionary where each key is a region id and the value a distribution collection.
    '''

    unique_trait_id = dict(zip([tuple(n.items()) for n in neurondb], range(len(neurondb))))

    morphs_by_layer = get_morphologies_by_layer(neurondb)

    regions_dists = {}

    for region_id, layer_ids in region_layers_map.items():
        region_morphs = get_morphologies_by_layer_group(morphs_by_layer, layer_ids)
        region_dist_table = get_placement_hints_table(region_morphs)
        morphs_ids = [unique_trait_id[tuple(m.items())] for m in region_morphs]

        for scores in region_dist_table.transpose():
            dist = dict(zip(morphs_ids, scores))
            dist = tt.normalize_probability_distribution(dist)
            regions_dists.setdefault(region_id, []).append(dist)

    return regions_dists


def assign_distributions_to_voxels(voxel_scores, dists):
    '''group voxels by a their score, and assign a distribution to each group.
    There will be as many groups as distributions. The distribution are assigned in order
    to the groups from the lowest scores to the higher scores

    Returns:
        An array of the same shape as voxel_scores, where each value is an index into
        the dists list
    '''
    count_per_bin, _ = np.histogram(voxel_scores, bins=max(len(dists), 1))
    voxel_indices = np.argsort(voxel_scores)

    region_dist_idxs = np.ones(shape=voxel_scores.shape, dtype=np.int) * -1
    region_dists = []

    i = 0
    for bin_count, dist in zip(count_per_bin, dists):
        indices = voxel_indices[i: i + bin_count]
        region_dist_idxs[indices] = len(region_dists)
        i += bin_count
        region_dists.append(dist)

    return region_dist_idxs


def transform_neurondb_into_spatial_distribution(annotation_raw, neurondb, region_layers_map):
    '''take the raw data from a neuron db (list of dicts) and build a volumetric dataset
    that contains the distributions of possible morphologies.

    Args:
        annotation_raw: voxel data from Allen Brain Institute to identify regions of space.
        neurondb: list of dicts containing the information extracted from a neurondb v4 file.
            only the 'layer' attribute is strictly needed
        region_layers_map: dict that contains the relationship between regions (referenced by
            the annotation) and layers (referenced by the neurondb). The keys are region ids
            and the values are tuples of layer ids.

    Returns:
        A SpatialDistribution object where the properties of the traits_collection are those
        obtained from the neurondb.
    '''

    # "outside" is tagged in the annotation_raw with 0
    # This will calculate, for every voxel, the euclidean distance to
    # the nearest voxel tagged as "outside" the brain
    distance_to_pia = distance_transform_edt(annotation_raw)

    # TODO take only the top 8% for each mtype-etype combination
    region_dists = get_region_distributions_from_placement_hints(neurondb, region_layers_map)

    flat_field = np.ones(shape=np.product(annotation_raw.shape), dtype=np.int) * -1

    all_dists = []

    for region_id, dists in region_dists.iteritems():
        voxel_distances = distance_to_pia[annotation_raw == region_id].flatten()
        voxel_dist_indices = assign_distributions_to_voxels(voxel_distances, dists)

        flat_mask = (annotation_raw == region_id).flatten()
        flat_field[flat_mask] = voxel_dist_indices + len(all_dists)

        all_dists.extend(dists)

    return tt.SpatialDistribution(flat_field.reshape(annotation_raw.shape),
                                  all_dists,
                                  neurondb)


def load_neurondb_v4_as_spatial_distribution(neurondb_filename,
                                             annotation_raw, hierarchy, region_name):
    '''load the bbp recipe and return a spatial voxel-based distribution
    returns: see transform_into_spatial_distribution
    '''
    region_layers_map = map_regions_to_layers(hierarchy, region_name)

    neurondb = load_neurondb_v4(neurondb_filename)

    return transform_neurondb_into_spatial_distribution(annotation_raw,
                                                        neurondb,
                                                        region_layers_map)
