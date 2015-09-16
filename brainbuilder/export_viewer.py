'''code to export all the intermediates and results of a new circuit for the JS viewer '''
import os
from os.path import join as joinp
from brainbuilder.utils import viewer


import logging
L = logging.getLogger(__name__)


def export_viewer(directory, voxel_dimensions,
                  positions,
                  orientation_field,
                  chosen_synapse_class, chosen_me, chosen_morphology):
    '''export all the intermediates and results of a new circuit for the JS viewer'''

    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError:
            L.exception('Need a directory to put circuit')

    # TODO consider downsampling points (for full brain, the viewer may crash)

    viewer.export_points(joinp(directory, 'mtype.pts'),
                         positions, 'mtype', [me[0] for me in chosen_me])

    viewer.export_points(joinp(directory, 'etype.pts'),
                         positions, 'etype', [me[1] for me in chosen_me])

    viewer.export_points(joinp(directory, 'sclass.pts'),
                         positions, 'sClass', chosen_synapse_class)

    for i in range(orientation_field.shape[-2]):
        viewer.export_vector_field(joinp(directory, 'field_%d.vcf' % i),
                                   orientation_field[..., i, :], 7000, voxel_dimensions)

    # TODO export chosen orientation for each point

    viewer.export_points(joinp(directory, 'morph.pts'),
                         positions, 'morphology', chosen_morphology)
