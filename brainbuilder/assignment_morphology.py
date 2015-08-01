'''algorithm to assign morphologies to a group of cells'''


# pylint: disable=W0613
def assign_morphology(positions, chosen_me, annotation, hierarchy,
                      recipe_filename, neurondb_filename):
    '''for every cell in positions, assign a morphology to each cell based on its metype
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        chosen_me: a list of metype values that correspond to each position
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        recipe_filename: BBP brain builder recipe
        neurondb_filename: neurondbV4.dat
    Returns:
        a list of morpholgies that correspond to each position
    '''
    # TODO do
    raise NotImplementedError
