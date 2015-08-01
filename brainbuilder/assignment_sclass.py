'''algorithm to assign me-types to a group of cells'''

from brainbuilder.utils.traits import assign_from_spacial_distribution


# pylint: disable=W0613
def load_type_distribution(filename):
    '''load an sclass spacial probability distribution'''
    # TODO do
    raise NotImplementedError


def assign_sclass(positions, sclass_distribution_filename, voxel_dimensions):
    '''for every cell in positions, chooses whether it's an Excitatory or Inhibitory neuron
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        sclass_distribution_filename: file that contains volume data where every voxel value
            represents a probability of a cell there of being Inhibitory
    Returns:
        a list of sclass values that correspond to each position
    '''
    field, probabilities = load_type_distribution(sclass_distribution_filename)
    return assign_from_spacial_distribution(positions, field, probabilities, voxel_dimensions)
