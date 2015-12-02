'''algorithm to compute orientation fields for Hippocampus'''
from brainbuilder.utils import genbrain as gb
from brainbuilder.utils import vector_fields as vf
from brainbuilder.select_region import select_hemisphere

from scipy.optimize import leastsq  # pylint: disable=E0611
import numpy as np


def leastsq_circle(x, y):
    '''fit a circle to a group of points'''

    def calculate_distances(_x, _y, _xc, _yc):
        '''calculate the distance of each 2D points from the center (xc, yc)'''
        return np.sqrt(np.square(_x - _xc) + np.square(_y - _yc))

    def fitness_function(c, _x, _y):
        '''calculate the algebraic distance between the data points and the mean circle'''
        ds = calculate_distances(_x, _y, *c)
        return ds - ds.mean()

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, _ = leastsq(fitness_function, center_estimate, args=(x, y))
    xc, yc = center
    distances = calculate_distances(x, y, *center)
    radius = distances.mean()
    residual = np.sum((distances - radius) ** 2)
    return xc, yc, radius, residual


def circular_tangent_field(x, y, xc, yc):
    '''return a group of normalized vectors at the given points that are tangents
    to the circles with the given centre '''
    dx = xc - x
    dy = yc - y

    distance = np.sqrt(np.square(dx) + np.square(dy))

    dx /= distance
    dy /= distance

    return -dy, dx


def compute_main_axis_hemispheric_field(mask, hemisphere):
    '''return a vector field covering only one hemisphere that represents the direction
    of the main axis of the hippocampus in that area'''
    idx = np.nonzero(select_hemisphere(mask, hemisphere))
    points = np.array(idx).transpose()

    yc_0, xc_0, _, _ = leastsq_circle(points[:, 1], points[:, 0])
    yc_1, zc_1, _, _ = leastsq_circle(points[:, 1], points[:, 2])

    # we use Y as the free variable and fit Z and X from it
    dz, dy = circular_tangent_field(points[:, 2], points[:, 1], zc_1, yc_1)
    dx, _ = circular_tangent_field(points[:, 0], points[:, 1], xc_0, yc_0)

    if hemisphere:
        # change the direciton of the tangents on the YX plane
        dx *= -1
    else:
        # TODO figure out what should be the symmetry convention for morphology placement
        dx *= -1
        dz *= -1
        dy *= -1

    tangents = np.array([dx, dy, dz]).transpose()

    dis = np.sqrt(np.sum(np.square(tangents), axis=-1))
    tangents /= dis[..., np.newaxis]

    tangents_field = np.zeros(shape=(mask.shape + (tangents.shape[1],)), dtype=np.float32)
    tangents_field[idx] = tangents

    return tangents_field


def compute_main_axis_field(mask):
    '''return a vector field that represents the direction
    of the main axis of the hippocampus'''

    left = compute_main_axis_hemispheric_field(mask, True)
    right = compute_main_axis_hemispheric_field(mask, False)
    return vf.join_vector_fields(left, right)


def compute_depth_axis_field(annotation, hierarchy, first, last, region_mask):
    '''return a vector field that represents the direction
    of the depth axis (across layers) of the hippocampus'''

    first_mask = gb.get_regions_mask_by_names(annotation.raw, hierarchy, first)
    last_mask = gb.get_regions_mask_by_names(annotation.raw, hierarchy, last)

    return vf.calculate_fields_by_distance_between(region_mask, first_mask, last_mask)


def compute_orientation_field(annotation, hierarchy, region_name):
    '''Computes the orientation field for the hippocampus

    Args:
        annotation: voxel data from Allen Brain Institute (can be crossrefrenced with hierarchy)
        hierarchy: json from Allen Brain Institute
        region_name: the exact name in the hierarchy that the field should be computed for

    Returns:
        A 5D numpy array of shape AxBxCx3x3 where AxBxC is the shape of annotation, the first
        dimension of size 3 differentiates between the right,up,forwards fields and the last
        dimension of size 3 contains the three i,j,k components of each vector

    '''
    subhierarchy = hierarchy.collect('name', region_name, 'id')
    region_mask = gb.get_regions_mask_by_ids(annotation.raw, subhierarchy)
    fwd_field = compute_main_axis_field(region_mask)

    first = [name for name in hierarchy.collect('name', region_name, 'name')
             if 'stratum lacunosum-moleculare' in name]
    last = [name for name in hierarchy.collect('name', region_name, 'name')
            if 'stratum oriens' in name]

    up_field = compute_depth_axis_field(annotation, hierarchy, first, last, region_mask)
    # the value of sigma is hand-picked to soften the edge errors we get on the Allen atlas
    up_field = vf.normalize(vf.gaussian_filter(up_field, sigma=2.5))

    right_field = np.cross(up_field, fwd_field)

    field = vf.combine_vector_fields([right_field, up_field, fwd_field])

    return gb.VoxelData(field, annotation.voxel_dimensions, annotation.offset)
