'''algorithm to clip a cell density distributions'''
import numpy as np
from voxcell import build


def select_region(annotation_raw, density, hierarchy, region_name, inverse=False):
    '''Trim a density voxel dataset to keep only those that belong to a desired region
    or the opposite. Returns a copy where regions of no interest have been clipped out.

    Args:
        annotation_raw: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        density: VoxelData object with voxel data from Allen Brain Institute.
            Called "atlasVolume" in their website.
            Each voxel represents a value that once normalised, can be treated as a probability
            of cells appearing in this voxel.
        hierarchy: json from Allen Brain Institute
        region_name: the name of the region of interest (can be crossrefrenced with hierarchy)
        inverse(bool): Invert the selection, so that everything *BUT* the region is selected

    Returns:
        density: exactly same format as input, but the values outside the region of interest has
        been set to 0
    '''
    in_region = build.mask_by_region_names(annotation_raw, hierarchy, [region_name])
    if inverse:
        in_region = np.invert(in_region)
    return density.with_data(density.raw * in_region)


def select_hemisphere(density_raw, left=True):
    '''Trim a density voxel dataset to keep only those in one of the two hemispheres.

    Args:
        density_raw: voxel data from Allen Brain Institute.
        left: if True select the left hemisphere (default), otherwise select the right one.
    '''
    in_region = np.ones_like(density_raw, dtype=np.bool)
    in_region[:, :, in_region.shape[2] / 2:] = False
    if not left:
        in_region = np.invert(in_region)
    return density_raw * in_region
