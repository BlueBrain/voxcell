'''algorithm to assign me-types to a group of cells'''


# pylint: disable=W0613
def assign_metype(positions, chosen_sclass, annotation, hierarchy, recipe_filename):
    '''for every cell in positions, assign me-type to each cell based on its  sclass
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        chosen_sclass: a list of sclass values that correspond to each position
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        recipe_filename: BBP brain builder recipe
    Returns:
        a list of me-type values that correspond to each position
    '''
    # TODO do
    raise NotImplementedError
