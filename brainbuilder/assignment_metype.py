'''algorithm to assign me-types to a group of cells'''
import numpy as np

from brainbuilder.utils import bbp
from brainbuilder.utils import traits as tt

import logging
L = logging.getLogger(__name__)


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
        for those positions whose me-type could not be determined, None is returned
    '''

    recipe_sdist = bbp.load_recipe_as_spatial_distributions(recipe_filename,
                                                            annotation.raw, hierarchy, region_name)

    subsections = tt.split_distribution_collection(recipe_sdist, 'sClass')

    chosen_metype = np.ones(shape=(len(chosen_sclass)), dtype=np.int) * -1

    for value, subdist in subsections.iteritems():
        mask = np.array(chosen_sclass) == value
        chosen_metype[mask] = tt.assign_from_spatial_distribution(positions[mask],
                                                                  subdist,
                                                                  annotation.mhd['ElementSpacing'])

    if np.count_nonzero(chosen_metype == -1):
        # this may happen becaues of inconsistencies of the data
        # for example if we assigned excitatory neurons to a neuron that is in
        # a voxel for which only inhibitory metype probabilities are known
        L.warning('%d / %d cells could not get a valid metype assigned',
                  np.count_nonzero(chosen_metype == -1), len(chosen_metype))

    return [{'mtype': recipe_sdist.traits[idx]['mtype'], 'etype': recipe_sdist.traits[idx]['etype']}
            if idx != -1 else None
            for idx in chosen_metype]
