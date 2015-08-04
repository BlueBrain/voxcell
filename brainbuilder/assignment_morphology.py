'''algorithm to assign morphologies to a group of cells'''

from brainbuilder.utils import traits as tt


# pylint: disable=W0613
def assign_morphology(positions, chosen_me, spatial_dist, voxel_dimensions):
    '''for every cell in positions, assign a morphology to each cell based on its metype
    Accepts:
        positions: list of positions for soma centers (x, y, z).
        chosen_me: a list of metype values that correspond to each position
        spatial_dist: SpatialDistribution containing at least the properties:
            mtype, etype, morphology
        voxel_dimensions: tuple with the size of the voxels in microns in each axis
    Returns:
        a list of morpholgies that correspond to each position
    '''
    # TODO take metype into account
    chosen_morphs = tt.assign_from_spatial_distribution(positions, spatial_dist, voxel_dimensions)
    return [spatial_dist.traits[idx]['morphology'] for idx in chosen_morphs]
