'''compatibility functions with existing BBP formats'''
import itertools
import logging
import numbers
import yaml
from six import iteritems
from six.moves import zip

import lxml.etree
import numpy as np
import pandas as pd

from voxcell import CellCollection, VoxelData
from voxcell import traits as tt
from voxcell.math_utils import lcmm, angles_to_matrices
from scipy.ndimage import distance_transform_edt  # pylint: disable=E0611

from brainbuilder.version import VERSION
from brainbuilder.exceptions import BrainBuilderError


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
    parser = lxml.etree.XMLParser(resolve_entities=False)
    return lxml.etree.parse(recipe_filename, parser=parser)


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
            layer, mtype, etype, morph_class, synapse_class, percentage
    '''
    recipe_tree = _parse_recipe(recipe_filename)

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
                                layer.attrib['id'],
                                structural_type.attrib['id'],
                                electro_type.attrib['id'],
                                structural_type.attrib['mClass'],
                                structural_type.attrib['sClass'],
                                percentage
                            ]

    return pd.DataFrame(read_records(), columns=['layer',
                                                 'mtype', 'etype',
                                                 'morph_class', 'synapse_class',
                                                 'percentage'])


def load_recipe_density(recipe_filename, annotation, region_layers_map):
    '''take a BBP builder recipe and return cell density VoxelData.

    Returns:
        VoxelData with cell density (expected cell count per voxel).
    '''
    recipe_tree = _parse_recipe(recipe_filename)

    percentages = dict((layer.attrib['id'], float(layer.attrib['percentage']) / 100)
                       for layer in recipe_tree.findall('NeuronTypes')[0].getchildren()
                       if layer.tag == 'Layer')

    raw = np.zeros_like(annotation.raw, dtype=np.float32)

    for rid, layers in iteritems(region_layers_map):
        assert len(layers) == 1
        if layers[0] in percentages:
            region_mask = annotation.raw == rid
            voxel_count = np.count_nonzero(region_mask)
            if voxel_count:
                raw[region_mask] = percentages[layers[0]] / float(voxel_count)
            else:
                L.warning('No voxels tagged for layer %s', layers[0])
        else:
            L.warning('No percentage found in recipe for layer %s', layers[0])

    total_neurons = int(recipe_tree.find('NeuronTypes').attrib['totalNeurons'])
    raw *= total_neurons

    return annotation.with_data(raw)


def transform_recipe_into_spatial_distribution(annotation, recipe, region_layers_map):
    '''take distributions grouped by layer ids and a map from regions to layers
    and build a volumetric dataset that contains the same distributions

    Returns:
        A SpatialDistribution object where the properties of the traits_collection are:
        mtype, etype, morph_class, synapse_class
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


def load_recipe_as_spatial_distribution(recipe_filename, annotation, region_layers_map):
    '''load the bbp recipe and return a spatial voxel-based distribution

    Returns:
        see transform_into_spatial_distribution
    '''
    recipe = get_distribution_from_recipe(recipe_filename)
    return transform_recipe_into_spatial_distribution(annotation, recipe, region_layers_map)


def _hex_prism_volume(side, height):
    return 3 * np.sqrt(3) * side * side * height / 2


def load_builder_recipe(recipe_filename, atlas, region_map):
    """
    Load BBP builder recipe.

    Returns:
        density: VoxelData with cell density (cell count / mm^3)
        sdist: (region, mtype, etype, morph_class, synapse_class) SpatialDistribution

    Density is calculated for 'column' defined in the recipe and mapped to `atlas`.
    """
    # pylint: disable=too-many-locals
    recipe_tree = _parse_recipe(recipe_filename)

    total_cell_count = int(recipe_tree.find('/NeuronTypes').attrib['totalNeurons'])
    layer_thickness = {
        elem.attrib['id']: float(elem.attrib['thickness'])
        for elem in recipe_tree.iterfind('/column/layer')
    }

    lattice = get_lattice_vectors(recipe_filename)
    a1 = np.linalg.norm(lattice['a1'])
    a2 = np.linalg.norm(lattice['a2'])
    assert np.isclose(a1, a2)
    hex_side = a1

    density = np.zeros_like(atlas.raw, dtype=np.float32)
    sdist_field = -1 * np.ones_like(atlas.raw, dtype=np.int16)
    dist_ids = {}

    for layer_id, layer_elem in enumerate(recipe_tree.iterfind('/NeuronTypes/Layer'), 1):
        layer = layer_elem.attrib['id']
        layer_ratio = 0.01 * float(layer_elem.attrib['percentage'])
        layer_height = layer_thickness[layer]
        layer_volume_mm3 = _hex_prism_volume(hex_side, layer_height) / 1e9
        layer_density = layer_ratio * total_cell_count / layer_volume_mm3
        L.debug("'%s' density: %.3f", layer, layer_density)
        region_ids = region_map.get(layer)
        if not region_ids:
            L.warning("No '%s' region in region map", layer)
            continue
        mask = np.isin(atlas.raw, list(region_ids))
        if not np.any(mask):
            L.warning("No voxels tagged for layer %s", layer)
            continue
        density[mask] = layer_density
        sdist_field[mask] = layer_id
        dist_ids[layer] = layer_id

    traits = get_distribution_from_recipe(recipe_filename)
    distributions = pd.DataFrame(
        data=0.0, index=traits.index, columns=sorted(dist_ids.values())
    )
    for layer, dist_id in iteritems(dist_ids):
        data = traits[traits.layer == layer]['percentage']
        distributions.loc[data.index, dist_id] = data.values
    distributions /= distributions.sum()

    traits.rename(columns={'layer': 'region'}, inplace=True)
    del traits['percentage']

    density = atlas.with_data(density)
    sdist = tt.SpatialDistribution(
        atlas.with_data(sdist_field), distributions, traits
    )
    return density, sdist


def load_recipe_cell_density(recipe_filename, atlas, region_map):
    """
    Take BBP cell recipe and return VoxelData with cell densities.

    TODO: link to recipe spec.

    Returns:
        VoxelData with cell densities (expected cell count per voxel).
    """
    recipe_tree = _parse_recipe(recipe_filename)

    result = np.zeros_like(atlas.raw, dtype=np.float32)

    for region in recipe_tree.iterfind('/Region'):
        mask = np.isin(atlas.raw, region_map[region.attrib['name']])
        if np.any(mask):
            result[mask] = float(region.attrib['density'])
        else:
            L.warning('No voxels tagged for region %s', region)

    return atlas.with_data(result)


def load_recipe_cell_traits(recipe_filename, atlas, region_map):
    """
    Take BBP cell recipe and return SpatialDistribution with cell traits.

    TODO: link to recipe spec.

    Returns:
        SpatialDistribution where the properties of the traits_collection are:
            region, mtype, etype, morph_class, synapse_class
    """
    def _parse_type(elem):
        etype_attr = elem.attrib
        mtype_attr = elem.getparent().attrib
        region_attr = elem.getparent().getparent().attrib
        return {
            'region': region_attr['name'],
            'mtype': mtype_attr['name'],
            'etype': etype_attr['name'],
            'morph_class': mtype_attr['mClass'],
            'synapse_class': mtype_attr['sClass'],
            'percentage': (
                (float(mtype_attr['percentage']) / 100.0) *
                (float(etype_attr['percentage']) / 100.0)
            ),
        }

    recipe_tree = _parse_recipe(recipe_filename)

    cell_traits = pd.DataFrame([
        _parse_type(elem) for elem in recipe_tree.iterfind('/Region/StructuralType/ElectroType')
    ])
    percentage = cell_traits.pop('percentage')

    distributions = pd.DataFrame(
        data=0.0,
        index=cell_traits.index,
        columns=np.unique(atlas.raw)
    )

    for region, region_ids in iteritems(region_map):
        data = percentage[cell_traits.region == region]
        for region_id in region_ids:
            distributions.loc[data.index, region_id] = data.values

    return tt.SpatialDistribution(atlas, distributions, cell_traits)


def load_metype_composition(filepath, atlas, region_map):
    """
    Load me-type composition defined as a set of mtype densities bound to atlas.

    Args:
        filepath: Path to YAML with me-type composition (TODO: link to spec)
        atlas: VoxelData with brain region IDs
        region_map: {<region> -> [IDs]} mapping

    Returns:
        total_density: VoxelData with total cell density
        sdist: SpatialDistribution of (region, mtype) properties
        etypes: dict of etype ratios per (region, mtype)

    Example YAML:
    >
      version: v1.0
      composition:
        L1:
          L1_HAC:
            density: HAC.nrrd
            density_factor: 0.9
            etypes:
              bNAC: 0.2
               cIR:  0.8
          L1_DAC:
                ...
        L2:
          L23_PC:
            density: 11000  # cells / mm^3
            etypes:
              cADpyr: 1.0
            ...
    """
    # pylint: disable=too-many-locals
    def _load_density(density, mask):
        """ Load density from NRRD or single float value + mask. """
        if isinstance(density, numbers.Number):
            result = np.zeros_like(mask, dtype=np.float32)
            result[mask] = float(density)
        else:
            result = VoxelData.load_nrrd(density).raw.astype(np.float32)
            result[~mask] = 0
        return result

    with open(filepath, 'r') as f:
        content = yaml.load(f)

    assert content['version'] == 'v1.0'
    composition = content['composition']

    densities, traits = [], []
    etypes = {}

    for region in sorted(composition.keys()):
        if region not in region_map:
            L.warning("Region '%s' not in the region map, skipping", region)
            continue
        region_mask = np.isin(atlas.raw, list(region_map[region]))
        if not np.any(region_mask):
            L.warning("No voxels found for region '%s'", region)
            continue
        for mtype in sorted(composition[region]):
            params = composition[region][mtype]
            L.info("Loading '%s' density...", mtype)
            density = _load_density(params['density'], region_mask)
            if 'density_factor' in params:
                density *= params['density_factor']
            densities.append(density)
            traits.append((region, mtype))
            etypes[(region, mtype)] = params['etypes']

    total_density = sum(densities)
    ijk = np.nonzero(total_density > 0)
    if len(ijk[0]) == 0:
        raise BrainBuilderError("No voxel with total density > 0")

    L.info("Composing (region, mtype) SpatialDistribution...")
    traits = pd.DataFrame(traits, columns=['region', 'mtype'])
    distributions = pd.DataFrame(
        np.stack(density[ijk] for density in densities) / total_density[ijk]
    )

    field = np.full_like(atlas.raw, -1, dtype=np.int32)
    field[ijk] = np.arange(distributions.shape[1])

    L.info("Done!")
    return (
        atlas.with_data(total_density),
        tt.SpatialDistribution(atlas.with_data(field), distributions, traits),
        etypes
    )


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
            yield {
                'morphology': fields[0],
                'layer': fields[1],
                'mtype': fields[2],
                'etype': fields[3],
                'me_combo': fields[4],
                'placement_hints': [float(x) for x in fields[5:]],
            }

    with open(neurondb_filename) as f:
        return pd.DataFrame(read_records(f.readlines()))


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
    subdivision_count = lcmm(morphs.placement_hints.map(len).as_matrix())

    region_dist_table = pd.DataFrame(dtype=np.float,
                                     index=morphs.index,
                                     columns=np.arange(subdivision_count))

    groups = morphs.placement_hints.groupby(lambda k: len(morphs.placement_hints[k]))
    for length, hints_group in groups:

        # TODO find a nicer way to get a 2D array from an array of lists
        count = len(hints_group)
        scores = np.array(list(itertools.chain(*hints_group.values))).reshape((count, length))

        # placement hints are organised bottom (high score) to top (low score)
        scores = np.fliplr(scores).astype(np.float)

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
    for k, v in iteritems(region_layers_map):
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
    for layer_ids, region_ids in iteritems(reverse_region_layers_map(region_layers_map)):

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
                      ', '.join(l for l in layer_ids))

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
        neurondb(dataframe):
            columns 'morphology', 'layer', 'mtype', 'etype', 'me_combo', 'placement_hints'
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
    for region_ids, dists in iteritems(region_dists):
        flat_mask = np.in1d(annotation.raw, region_ids)

        voxel_distances = metric[flat_mask]
        voxel_dist_indices = assign_distributions_to_voxels(voxel_distances, len(dists.columns))

        offset = len(all_dists.columns)
        dists.columns += offset
        flat_field[flat_mask] = voxel_dist_indices + offset
        all_dists = pd.concat([all_dists, dists], axis=1) # pylint: disable=redefined-variable-type

    field = annotation.with_data(flat_field.reshape(annotation.raw.shape))

    return tt.SpatialDistribution(field, all_dists.fillna(0.0), neurondb)


def get_distance_to_pia(annotation):
    '''given an atlas, compute a voxel dataset of the same shape where every voxel value
    represents the distance to the pia region'''

    # "outside" is tagged in the annotation_raw with 0
    # This will calculate, for every voxel, the euclidean distance to
    # the nearest voxel tagged as "outside" the brain
    return distance_transform_edt(annotation.raw)


def load_neurondb_v4_as_spatial_distribution(neurondb_filename,
                                             annotation, region_layers_map,
                                             percentile):
    '''load the bbp recipe and return a spatial voxel-based distribution

    Returns:
        see transform_into_spatial_distribution
    '''
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
            ('x', float), ('y', float), ('z', float), ('r', float), ('me_combo', str)
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


def _matrices_to_angles(matrices):
    """
    Convert 3x3 rotation matrices to rotation angles around Y.

    Use NaN if rotation could not be represented as a single rotation angle.
    """
    phi = np.arccos(matrices[:, 0, 0]) * np.sign(matrices[:, 0, 2])
    mat = angles_to_matrices(phi, axis='y')
    valid = np.all(np.isclose(mat, matrices), axis=(1, 2))
    phi[~valid] = np.nan
    return phi


def load_mvd2(filepath):
    '''loads an mvd2 as a CellCollection'''
    data = parse_mvd2(filepath)

    cells = CellCollection()

    cells.positions = np.array([[c['x'], c['y'], c['z']] for c in data['Neurons Loaded']])

    angles = np.array([c['r'] for c in data['Neurons Loaded']]) * np.pi / 180
    cells.orientations = angles_to_matrices(angles, axis='y')

    props = pd.DataFrame({
        'synapse_class': [data['MorphTypes'][c['mtype']]['sclass']
                          for c in data['Neurons Loaded']],
        'morph_class': [data['MorphTypes'][c['mtype']]['mclass'] for c in data['Neurons Loaded']],
        'mtype': [data['MorphTypes'][c['mtype']]['name'] for c in data['Neurons Loaded']],
        'etype': [data['ElectroTypes'][c['etype']]['name'] for c in data['Neurons Loaded']],
        'morphology': [c['morphology'] for c in data['Neurons Loaded']],
        'layer': [str(1 + c['layer']) for c in data['Neurons Loaded']],
        'hypercolumn': [c['hyperColumn'] for c in data['Neurons Loaded']],
        'minicolumn': [c['miniColumn'] for c in data['Neurons Loaded']],
        'me_combo': [c['me_combo'] for c in data['Neurons Loaded']],
    })

    cells.add_properties(props)
    return cells


def save_mvd2(filepath, morphology_path, cells):
    '''saves a CellCollection as mvd2

    Rotations might be lost in the process.
    Cells are expected to have the properties:
    'morphology', 'mtype', 'etype', 'morph_class', 'synapse_class', 'me_combo';
    and, optionally, 'hypercolumn', 'minicolumn', 'layer'.
    '''
    # pylint: disable=too-many-locals
    rotations = 180 * _matrices_to_angles(cells.orientations) / np.pi
    if np.count_nonzero(np.isnan(rotations)):
        L.warning("save_mvd2: some rotations would be lost!")

    optional = {}
    for prop in ('hypercolumn', 'minicolumn', 'layer'):
        if prop in cells.properties:
            optional[prop] = cells.properties[prop]
        else:
            L.warning("save_mvd2: %s not specified, zero will be used", prop)
            optional[prop] = np.zeros(len(cells.properties), dtype=np.int)

    electro_types, chosen_etype = np.unique(cells.properties.etype, return_inverse=True)

    mtype_names, chosen_mtype = np.unique(cells.properties.mtype, return_inverse=True)

    morph_types = []
    for mtype_name in mtype_names:
        mask = (cells.properties.mtype == mtype_name).values
        morph_types.append((mtype_name,
                            cells.properties[mask].morph_class.values[0],
                            cells.properties[mask].synapse_class.values[0]))

    def get_mvd2_neurons():
        '''return the data for all the neurons used in the circuit'''
        data = zip(
            cells.properties.morphology,
            cells.positions,
            rotations,
            chosen_mtype,
            chosen_etype,
            optional['hypercolumn'],
            optional['minicolumn'],
            optional['layer'],
            cells.properties.me_combo,
        )

        for morph, pos, phi, mtype_idx, etype_idx, hypercolumn, minicolumn, layer, me_combo in data:
            yield dict(name=morph, mtype_idx=mtype_idx, etype_idx=etype_idx,
                       rotation=phi, x=pos[0], y=pos[1], z=pos[2], hypercolumn=hypercolumn,
                       minicolumn=minicolumn, layer=int(layer) - 1, me_combo=me_combo)

    with open(filepath, 'w') as fd:
        fd.write("Application:'BrainBuilder {version}'\n"
                 "{morphology_path}\n"
                 "/unknown/\n".format(version=VERSION, morphology_path=morphology_path))

        fd.write('Neurons Loaded\n')
        line = ('{name} {database} {hypercolumn} {minicolumn} {layer} {mtype_idx} '
                '{etype_idx} {x} {y} {z} {rotation} {me_combo}\n')
        fd.writelines(line.format(database=0, **c) for c in get_mvd2_neurons())

        # skipping sections:
        # MicroBox Data
        # MiniColumnsPosition
        # CircuitSeeds

        fd.write('MorphTypes\n')
        fd.writelines('%s %s %s\n' % m for m in morph_types)

        fd.write('ElectroTypes\n')
        fd.writelines('%s\n' % e for e in electro_types)


def gid2str(gid):
    """ 42 -> 'a42' """
    return "a%d" % gid


def write_target(f, name, gids=None, include_targets=None):
    """ Append contents to .target file. """
    f.write("\nTarget Cell %s\n{\n" % name)
    if gids is not None:
        f.write("  ")
        f.write(" ".join(map(gid2str, gids)))
        f.write("\n")
    if include_targets is not None:
        f.write("  ")
        f.write(" ".join(include_targets))
        f.write("\n")
    f.write("}\n")


def write_property_targets(f, cells, prop, mapping=None):
    """ Append targets based on 'prop' cell property to .target file. """
    for value, gids in sorted(iteritems(cells.groupby(prop).groups)):
        if mapping is not None:
            value = mapping(value)
        write_target(f, value, gids=gids)
