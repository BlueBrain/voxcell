'''compatibility functions with existing BBP formats'''
import itertools
import xml.etree.ElementTree
from collections import defaultdict
from itertools import izip
import logging

import numpy as np
import pandas as pd
from voxcell import core
from voxcell import math
from voxcell import traits as tt
from scipy.ndimage import distance_transform_edt  # pylint: disable=E0611
from brainbuilder.version import VERSION

L = logging.getLogger(__name__)


def map_regions_to_layers(hierarchy, region_name):
    '''map regions in the hierarchy to the layer divisions used in BBP

    Returns:
        A dictionary where the key is the region id according to the Allen Brain data and
        the value is a tuple of the integer indices of the 6 layers used in BBP: 1, 2, 3, 4, 5, 6
    '''

    sub_area_names = hierarchy.collect('name', region_name, 'name')

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
                area = hierarchy.find('name', subarea)
                layer_groups[area[0].data['id']] = indices

    return layer_groups


def _parse_recipe(recipe_filename):
    '''parse a BBP recipe and return the corresponding etree'''
    # This is rather hacky but lets us avoid a dependency on lxml (which requires libxml)
    # the only reason to use lxml is because of the recipe uses the SYSTEM ELEMENT feature
    # that embeds one xml into another xml file, which is not supported on the standard xml
    # package
    # However, this is only used to embed the connectivy recipe into the rest of the recipe and
    # brain builder doesn't care about connectivity rules
    # This will skip unknown entities (normal dict would raise KeyError instead)
    parser = xml.etree.ElementTree.XMLParser()
    parser.entity = defaultdict(lambda: '')
    return xml.etree.ElementTree.parse(recipe_filename, parser=parser)


def get_lattice_vectors(recipe_filename):
    ''' get lattice vectors from recipe '''
    r = _parse_recipe(recipe_filename)
    lattice_vectors = r.find('column').findall('latticeVector')

    def get_lattice_by_id(lattice_vectors, id_lv):
        ''' search for the first lattice that has attribute "id" being id_lv '''
        return next(lv for lv in lattice_vectors if lv.attrib['id'] == id_lv)

    a1 = get_lattice_by_id(lattice_vectors, 'a1')
    a2 = get_lattice_by_id(lattice_vectors, 'a2')

    return {
        'a1': np.array([float(a1.attrib['x']), float(a1.attrib['z'])]),
        'a2': np.array([float(a2.attrib['x']), float(a2.attrib['z'])])
    }


def get_layer_thickness(recipe_filename):
    ''' get a map  id of the layer to their thickness '''
    r = _parse_recipe(recipe_filename)
    layers = r.find('column').findall('layer')
    return dict((int(l.attrib['id']), float(l.attrib['thickness'])) for l in layers)


def get_total_neurons(recipe_filename):
    ''' get the total number of neurons according to the recipe '''
    r = _parse_recipe(recipe_filename)
    total_neurons = r.find('NeuronTypes').attrib['totalNeurons']
    return int(total_neurons)


def get_distribution_from_recipe(recipe_filename):
    '''take a BBP builder recipe and return the probability distributions for each type

    Returns:
        A DataFrame with one row for each possibility and columns:
            layer, mtype, etype, mClass, sClass, percentage
    '''
    recipe_tree = _parse_recipe(recipe_filename)

    synapse_class_alias = {
        'INH': 'inhibitory',
        'EXC': 'excitatory'
    }

    def read_records():
        '''parse each neuron posibility in the recipe'''

        for layer in recipe_tree.findall('NeuronTypes')[0].getchildren():

            for structural_type in layer.getchildren():
                if structural_type.tag == 'StructuralType':

                    for electro_type in structural_type.getchildren():
                        if electro_type.tag == 'ElectroType':

                            percentage = (float(structural_type.attrib['percentage']) / 100 *
                                          float(electro_type.attrib['percentage']) / 100 *
                                          float(layer.attrib['percentage']) / 100)

                            yield [
                                int(layer.attrib['id']),
                                structural_type.attrib['id'],
                                electro_type.attrib['id'],
                                structural_type.attrib['mClass'],
                                synapse_class_alias[structural_type.attrib['sClass']],
                                percentage
                            ]

    return pd.DataFrame(read_records(), columns=['layer',
                                                 'mtype', 'etype',
                                                 'mClass', 'synapse_class',
                                                 'percentage'])


def load_recipe_density(recipe_filename, annotation, region_layers_map):
    '''take a BBP builder recipe and return the probability distributions for each type

    Returns:
        A DataFrame with one row for each posibility and columns:
            layer, mtype, etype, mClass, sClass, percentage
    '''
    recipe_tree = _parse_recipe(recipe_filename)

    percentages = dict((int(layer.attrib['id']), float(layer.attrib['percentage']) / 100)
                       for layer in recipe_tree.findall('NeuronTypes')[0].getchildren()
                       if layer.tag == 'Layer')

    raw = np.zeros_like(annotation.raw, dtype=np.float32)

    for rid, layers in region_layers_map.iteritems():
        assert len(layers) == 1
        if layers[0] in percentages:
            region_mask = annotation.raw == rid
            voxel_count = np.count_nonzero(region_mask)
            if voxel_count:
                raw[region_mask] = percentages[layers[0]] / float(voxel_count)
            else:
                L.warning('No voxels tagged for layer %d', layers[0])
        else:
            L.warning('No percentage found in recipe for layer %d', layers[0])

    raw /= np.sum(raw)

    return core.VoxelData(raw, annotation.voxel_dimensions, annotation.offset)


def transform_recipe_into_spatial_distribution(annotation, recipe, region_layers_map):
    '''take distributions grouped by layer ids and a map from regions to layers
    and build a volumetric dataset that contains the same distributions

    Returns:
        A SpatialDistribution object where the properties of the traits_collection are:
        mtype, etype, mClass, sClass
    '''
    distributions = pd.DataFrame(data=0.0,
                                 index=recipe.index,
                                 columns=region_layers_map.keys())

    for region_id, layer_ids in region_layers_map.items():
        for layer_id in layer_ids:
            data = recipe[recipe.layer == layer_id]['percentage']
            distributions.loc[data.index, region_id] = data.values

    distributions /= distributions.sum()

    return tt.SpatialDistribution(annotation, distributions, recipe)


def load_recipe_as_spatial_distribution(recipe_filename, annotation, hierarchy, region_name):
    '''load the bbp recipe and return a spatial voxel-based distribution

    Returns:
        see transform_into_spatial_distribution
    '''
    region_layers_map = map_regions_to_layers(hierarchy, region_name)

    recipe = get_distribution_from_recipe(recipe_filename)

    return transform_recipe_into_spatial_distribution(annotation,
                                                      recipe,
                                                      region_layers_map)


def load_neurondb_v4(neurondb_filename):
    '''load a neurondb v4 file

    Returns:
        A DataFrame where the columns are:
            morphology, layer, mtype, etype, metype, placement_hints
    '''

    def read_records(lines):
        '''parse each record in a neurondb file'''
        for line in lines:
            if not line.strip():
                continue
            fields = line.split()
            morphology, layer, mtype, etype, _ = fields[:5]
            placement_hints = list(float(h) for h in fields[5:])
            # skipping metype because it's just a combination of the mtype and etype values
            yield [morphology, int(layer), mtype, etype, placement_hints]

    with open(neurondb_filename) as f:
        return pd.DataFrame(read_records(f.readlines()),
                            columns=['morphology', 'layer', 'mtype', 'etype', 'placement_hints'])


def get_morphologies_by_layer(neurondb):
    '''group morphologies by layer

    Args:
        neurondb: A DataFrame with the contents of a neurondbv4.dat (see load_neurondb_4).

    Returns:
        A dictionary where the keys are layer ids and the values lists of morphologies
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


def clip_columns_to_percentile(dists, percentile):
    '''Clip distribution to, by region (ie: column)

    Args:
        dists(DataFrame): each row is a morphology, each column is a region
        percentile(float): percentile above which the morphologies are used, below which, their
            probability is set to 0 (ie: not used) range from (0.0, 1.0]
    '''
    # TODO: Should be albe to do this without iterating across columns
    for col_name in dists:
        dist = dists[col_name]
        percentile_value = np.percentile(dist, q=100 * percentile)
        dist[dist < percentile_value] = 0.0
    return dists


def get_placement_hints_table(morphs):
    '''collect the placement hint scores for a group of morphologies.

    Placement hints are a series of numbers associated with each morphology. This numbers
    represent how good a fit a morphology is to each subsection of space the space is evenly
    divided

    For example, having a morphology with scores [1, 2, 1] means that it is more likely to
    find this morphology in the second third of a space than it is to find it in the first or
    the last thirds.

    The original concept of "space" was layers and they were divided in the Y direction
    (towards pia). This allowed, for example, having morphologies appear only in the bottom
    half of a layer. Now that we are dealing with complex volumes, bottom and top don't mean
    much. Here "space" is just a collection of voxels which can be grouped according to some
    metric (distance to exterior).

    Note that this metric is applied to the voxel bins in reverse order because the placement
    hints are sorted bottom to top which means biggest distance to smallest distance.

    See BlueBuilder function:TinterfaceLayer::createMicrocircuitColumn
    in Objects/interfaceLayer.cxx: 717

    Args:
        morphs: a collection of morphologies.

    Returns:
        A DataFrame array that contains the placement hint scores for the given morphologies.
        This table has one row for each morphology and one column for each region subdivision
    '''
    subdivision_count = math.lcmm(morphs.placement_hints.map(len).as_matrix())

    region_dist_table = pd.DataFrame(dtype=np.float,
                                     index=morphs.index,
                                     columns=np.arange(subdivision_count))

    groups = morphs.placement_hints.groupby(lambda k: len(morphs.placement_hints[k]))
    for length, hints_group in groups:

        # TODO find a nicer way to get a 2D array from an array of lists
        count = len(hints_group)
        scores = np.array(list(itertools.chain(*hints_group.values))).reshape((count, length))

        # placement hints are organised bottom (high score) to top (low score)
        scores = np.fliplr(scores)

        repetitions = [subdivision_count // length] * length
        extended = np.repeat(scores, repetitions, axis=1)

        region_dist_table.ix[hints_group.index] = extended

    return region_dist_table


def reverse_region_layers_map(region_layers_map):
    ''' reverse the mapping between layers and regions

    Args:
        region_layers_map: a dict where the keys are region ids and the values tuples of layer ids

    Returns:
        A dict where the keys are tuples of layer ids and the keys lists of region ids'''
    inv_map = {}
    for k, v in region_layers_map.iteritems():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)

    return inv_map


def get_region_distributions_from_placement_hints(neurondb, region_layers_map, percentile):
    '''for every region, return the list of probability distributions for each potential
    morphology. The probabilites are taken from the placement hint scores.
    There is one distribution for each subdivision of the region and they are sorted
    the same way as the placement hint scores are: from furthest to pia to closest to pia

    Args:
        neurondb(dataframe): columns 'morphology', 'layer', 'mtype', 'etype', 'placement_hints'
        region_layers_map: dict that contains the relationship between regions (referenced by
            the annotation) and layers (referenced by the neurondb). The keys are region ids
            and the values are tuples of layer ids.
        percentile(float): percentile above which the morphologies are used, below which, their
            probability is set to 0 (ie: not used)

    Returns:
        A dict where each key is a tuple of region ids and the value a distribution collection.
    '''

    regions_dists = {}
    for layer_ids, region_ids in reverse_region_layers_map(region_layers_map).iteritems():

        mask = np.in1d(neurondb.layer, layer_ids)
        if np.any(mask):
            region_morphs = neurondb[mask].copy()

            dists = get_placement_hints_table(region_morphs)

            me_groups = region_morphs.groupby(['mtype', 'etype'])
            for idx in me_groups.indices.values():
                dists.iloc[idx] = clip_columns_to_percentile(dists.iloc[idx].copy(), percentile)

            regions_dists[tuple(region_ids)] = dists / dists.sum()
        else:
            L.warning('Layer %s from region-layer map not found in neurondb',
                      ', '.join(str(l) for l in layer_ids))

    return regions_dists


def assign_distributions_to_voxels(voxel_scores, bins):
    '''group voxels by a their score, and assign a distribution to each group.
    There will be as many groups as distributions. The distributions are assigned in order
    to the groups from the lowest scores to the higher scores

    Returns:
        An array of the same shape as voxel_scores, where each value is an index
        in the interval [0, bins)
    '''
    count_per_bin, _ = np.histogram(voxel_scores, bins=max(bins, 1))
    voxel_indices = np.argsort(voxel_scores)

    region_dist_idxs = np.ones(shape=voxel_scores.shape, dtype=np.int) * -1

    idx = 0
    for dist_idx, bin_count in enumerate(count_per_bin):
        indices = voxel_indices[idx: idx + bin_count]
        region_dist_idxs[indices] = dist_idx
        idx += bin_count

    return region_dist_idxs


def transform_neurondb_into_spatial_distribution(annotation, neurondb, region_layers_map,
                                                 metric,
                                                 percentile):
    '''take the raw data from a neuron db (list of dicts) and build a volumetric dataset
    that contains the distributions of possible morphologies.

    In the context of layers, the bins for the placement hint are numbered from the bottom of
    the layer (further from pia) towards the top (closer to pia)

    Args:
        annotation: voxel data from Allen Brain Institute to identify regions of space.
        neurondb(dataframe): columns 'morphology', 'layer', 'mtype', 'etype', 'placement_hints'
        region_layers_map: dict that contains the relationship between regions (referenced by
            the annotation) and layers (referenced by the neurondb). The keys are region ids
            and the values are tuples of layer ids.
        metric(np.array): numpy array with the same shape as annotation where each value is the
            value to map to the placement hints.
        percentile(float): percentile above which the morphologies are used, below which, their
            probability is set to 0 (ie: not used) range from (0.0, 1.0]

    Returns:
        A SpatialDistribution object where the properties of the traits_collection are those
        obtained from the neurondb.
    '''
    assert metric.shape == annotation.raw.shape

    metric = metric.flatten()

    region_dists = get_region_distributions_from_placement_hints(neurondb, region_layers_map,
                                                                 percentile)

    flat_field = np.ones(shape=np.product(annotation.raw.shape), dtype=np.int) * -1

    all_dists = pd.DataFrame()

    for region_ids, dists in region_dists.iteritems():
        flat_mask = np.in1d(annotation.raw, region_ids)

        voxel_distances = metric[flat_mask]
        voxel_dist_indices = assign_distributions_to_voxels(voxel_distances, len(dists.columns))

        offset = len(all_dists.columns)
        dists.columns += offset
        flat_field[flat_mask] = voxel_dist_indices + offset
        all_dists = pd.concat([all_dists, dists], axis=1)

    field = core.VoxelData(flat_field.reshape(annotation.raw.shape),
                           annotation.voxel_dimensions,
                           annotation.offset)

    return tt.SpatialDistribution(field, all_dists.fillna(0.0), neurondb)


def get_distance_to_pia(annotation):
    '''given an atlas, compute a voxel dataset of the same shape where every voxel value
    represents the distance to the pia region'''

    # "outside" is tagged in the annotation_raw with 0
    # This will calculate, for every voxel, the euclidean distance to
    # the nearest voxel tagged as "outside" the brain
    return distance_transform_edt(annotation.raw)


def load_neurondb_v4_as_spatial_distribution(neurondb_filename,
                                             annotation, hierarchy, region_name,
                                             percentile):
    '''load the bbp recipe and return a spatial voxel-based distribution

    Returns:
        see transform_into_spatial_distribution
    '''
    region_layers_map = map_regions_to_layers(hierarchy, region_name)

    neurondb = load_neurondb_v4(neurondb_filename)

    return transform_neurondb_into_spatial_distribution(annotation,
                                                        neurondb,
                                                        region_layers_map,
                                                        get_distance_to_pia(annotation),
                                                        percentile)


def parse_mvd2(filepath):
    '''loads an mvd2 as a dict data structure with tagged fields'''

    sections = {
        'Neurons Loaded': (
            ('morphology', str),
            ('database', int), ('hyperColumn', int), ('miniColumn', int),
            ('layer', int), ('mtype', int), ('etype', int),
            ('x', float), ('y', float), ('z', float), ('r', float), ('metype', str)
        ),
        'MicroBox Data': (
            ('size_x', float), ('size_y', float), ('size_z', float),
            ('layer_6_percentage', float),
            ('layer_5_percentage', float),
            ('layer_4_percentage', float),
            ('layer_3_percentage', float),
            ('layer_2_percentage', float)
        ),
        'MiniColumnsPosition': (('x', float), ('y', float), ('z', float)),
        'CircuitSeeds': (('RecipeSeed', float), ('ColumnSeed', float), ('SynapseSeed', float)),
        'MorphTypes': (('name', str), ('mclass', str), ('sclass', str)),
        'ElectroTypes': (('name', str),),
    }

    result = {}

    section_names = dict((s.lower(), s) for s in sections.keys())

    current_section = 'HEADER'

    with open(filepath) as f:
        for exact_line in f.readlines():
            line = exact_line.strip()

            if line.lower() in section_names:
                current_section = section_names[line.lower()]
            else:
                if current_section in sections:
                    fields = sections[current_section]
                    parsed = dict((field_def[0], field_def[1](value))
                                  for field_def, value in zip(fields, line.split()))

                    result.setdefault(current_section, []).append(parsed)
                else:
                    assert current_section == 'HEADER'
                    result.setdefault(current_section, '')
                    result[current_section] += exact_line + '\n'

    return result


def load_mvd2(filepath):
    '''loads an mvd2 as a CellCollection'''
    data = parse_mvd2(filepath)

    cells = core.CellCollection()

    cells.positions = np.array([[c['x'], c['y'], c['z']] for c in data['Neurons Loaded']])

    angles = np.array([c['r'] for c in data['Neurons Loaded']])
    cells.orientations = np.array([[[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
                                   for cos, sin in zip(np.cos(angles), np.sin(angles))])

    synapse_class_alias = {'INH': 'inhibitory', 'EXC': 'excitatory'}

    props = pd.DataFrame({
        'synapse_class': [synapse_class_alias[data['MorphTypes'][c['mtype']]['sclass']]
                          for c in data['Neurons Loaded']],
        'morph_class': [data['MorphTypes'][c['mtype']]['mclass'] for c in data['Neurons Loaded']],
        'mtype': [data['MorphTypes'][c['mtype']]['name'] for c in data['Neurons Loaded']],
        'etype': [data['ElectroTypes'][c['etype']]['name'] for c in data['Neurons Loaded']],
        'morphology': [c['morphology'] for c in data['Neurons Loaded']],
        'layer': [c['layer'] for c in data['Neurons Loaded']],
        'minicolumn': [c['miniColumn'] for c in data['Neurons Loaded']],
        'metype': [c['metype'] for c in data['Neurons Loaded']],
    })

    cells.add_properties(props)
    return cells


def save_mvd2(filepath, morphology_path, cells):
    '''saves a CellCollection as mvd2

    Rotations are lost in the process.
    Cells are expected to have the properties:
    morphology, mtype, etype, minicolumn, layer, morph_class and synapse_class
    '''

    map_exc_inh = {
        'excitatory': 'EXC',
        'inhibitory': 'INH',
    }

    electro_types, chosen_etype = np.unique(cells.properties.etype, return_inverse=True)

    mtype_names, chosen_mtype = np.unique(cells.properties.mtype, return_inverse=True)

    morph_types = []
    for mtype_name in mtype_names:
        mask = (cells.properties.mtype == mtype_name).values
        morph_types.append((mtype_name,
                            cells.properties[mask].morph_class.values[0],
                            map_exc_inh[cells.properties[mask].synapse_class.values[0]]))

    def get_mvd2_neurons():
        '''return the data for all the neurons used in the circuit'''
        data = izip(cells.properties.morphology,
                    cells.positions,
                    chosen_mtype,
                    chosen_etype,
                    cells.properties.minicolumn,
                    cells.properties.layer,
                    cells.properties.metype)

        for morph, pos, mtype_idx, etype_idx, minicolumn, layer, metype in data:
            yield dict(name=morph, morphology=mtype_idx, electrophysiology=etype_idx,
                       rotation=0.0, x=pos[0], y=pos[1], z=pos[2],
                       minicolumn=minicolumn, layer=layer, metype=metype)

    with open(filepath, 'w') as fd:
        fd.write("Application:'BrainBuilder {version}'\n"
                 "{morphology_path}\n"
                 "/unknown/\n".format(version=VERSION, morphology_path=morphology_path))

        fd.write('Neurons Loaded\n')
        line = ('{name} {database} {hyperColumn} {minicolumn} {layer} {morphology} '
                '{electrophysiology} {x} {y} {z} {rotation} {metype}\n')
        fd.writelines(line.format(database=0, hyperColumn=0, **c) for c in get_mvd2_neurons())

        # skipping sections:
        # MicroBox Data
        # MiniColumnsPosition
        # CircuitSeeds

        fd.write('MorphTypes\n')
        fd.writelines('%s %s %s\n' % m for m in morph_types)

        fd.write('ElectroTypes\n')
        fd.writelines('%s\n' % e for e in electro_types)
