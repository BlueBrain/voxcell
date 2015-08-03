'''algorithm to assign me-types to a group of cells'''

from brainbuilder.utils import bbp
from brainbuilder.utils import traits


# pylint: disable=W0613
def assign_metype(positions, chosen_sclass, annotation, hierarchy, recipe_filename, region_name):
    '''for every cell in positions, assign me-type to each cell based on its  synapse class
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        chosen_sclass: a list of synapse class values that correspond to each position
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        recipe_filename: BBP brain builder recipe
    Returns:
        a list of me-type values that correspond to each position
    '''

    (traits_field, probabilities, traits_collection) = \
        bbp.load_recipe_as_spatial_distributions(recipe_filename,
                                                 annotation.raw, hierarchy, region_name)

    # TODO trim distributions based on chosen_synapse_class
    assigned = traits.assign_from_spacial_distribution(positions,
                                                       traits_field, probabilities,
                                                       annotation.mhd['ElementSpacing'])

    return [{'mtype': traits_collection[idx]['mtype'],
             'etype': traits_collection[idx]['etype']}
            for idx in assigned]
