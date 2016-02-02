
# coding: utf-8

# In[1]:

from collections import namedtuple
import fractions
import numpy as np
import pandas as pd
from voxcellview.widget import VoxcellWidget


from voxcell.core import CellCollection, VoxelData
from voxcell import build

get_ipython().magic(u'matplotlib inline')


# network properties from: https://collab.humanbrainproject.eu/#/collab/375/nav/3533

# units for length (heights, widths,...) are all in µm.

# # density

# ## layered annotations

# In[2]:

LayerDefinition = namedtuple("LayerDefinition", "name height id")
layer_definitions = [
    # Granular Layer
    LayerDefinition("GL", 151, 1),
    # Purkinje cell Layer
    LayerDefinition("PL", 35, 2),
    # Molecular Layer
    LayerDefinition("ML", 300, 3),
]

volume_length = 600
volume_width = 600


# In[3]:

layer_ids = [l.id for l in reversed(layer_definitions)]
layer_heights = [l.height for l in reversed(layer_definitions)]


# In[4]:

def get_voxel_side(layer_heights):
    rounded_heights = np.round(layer_heights)
    result = rounded_heights[0]
    for n in rounded_heights:
        result = fractions.gcd(result, n)
    result = max(5, result)
    return result

voxel_side = get_voxel_side(layer_heights)

layer_heights_voxel = np.round(np.array(layer_heights) / voxel_side).astype(np.uint)
volume_length_voxel = volume_length / voxel_side
volume_width_voxel = volume_width / voxel_side


# In[5]:

annotation = VoxelData(build.layered_annotation((volume_length_voxel, volume_width_voxel),
                                                layer_heights_voxel, layer_ids),
                       [voxel_side] * 3)


# In[6]:

VoxcellWidget().show_volume('annotation', annotation, {"particle_size": 3})


# ## layer densities

# In[7]:

# unit is cells per µ^3
# original data are cells per mm^3
DENSITY_RATIO = 1e-9  # to convert between mm^3 and micron^3

density = {
    "GL": {
        "Glomeruli": 3e5 * DENSITY_RATIO,
        # Golgi Cells
        "GoC": 9e3 * DENSITY_RATIO,
        # Granule Cells
        "GrC": 4e6 * DENSITY_RATIO,
    },
    "PL": {
        # Purkinje Cells
        "PC": 6e5 * DENSITY_RATIO,
    },
    "ML": {
        # Stellate Cells
        "SC": 1e5 * DENSITY_RATIO,
        # Basket Cells
        "BC": 1e5 * DENSITY_RATIO,
    },
}
d_density = pd.DataFrame()


# In[8]:

id_to_layer_name = dict((ld.id, ld.name) for ld in layer_definitions)


# In[9]:

layer_name_to_id = dict((ld.name, ld.id) for ld in layer_definitions)


# In[10]:

voxel_raw = np.zeros_like(annotation.raw, dtype=np.float32)


# In[11]:

for l_id in layer_ids:
    mask = annotation.raw == l_id
    voxel_count = np.count_nonzero(mask)
    layer_name = id_to_layer_name[l_id]
    voxel_raw[mask] = float(sum(density[layer_name].values()))

voxel_density = VoxelData(voxel_raw, annotation.voxel_dimensions, annotation.offset)


# In[12]:

VoxcellWidget().show_volume('density', voxel_density, {"particle_size": 3})


# # cell positions

# In[13]:

layer_volumes = dict((layer.name, layer.height * volume_length * volume_width)
                     for layer in layer_definitions)


# In[14]:

layer_volumes


# In[15]:

total_cell_count = round(sum(layer_volumes[layer] * d
                             for layer in layer_volumes
                             for d in density[layer].values()))


# In[16]:

total_cell_count


# In[17]:

from brainbuilder.cell_positioning import cell_positioning


# In[18]:

new_cells = CellCollection()


# In[19]:

new_cells.positions = cell_positioning(voxel_density, int(total_cell_count))


# In[20]:

VoxcellWidget().show_points('position', new_cells, {"particle_size": 1})


# # mtype assignment

# ## traits

# In[21]:

traits_mtype = pd.DataFrame([[mtype] for layer in density
                             for mtype in density[layer]],
                            columns=["mtype"])
traits_mtype


# ## distribution

# In[22]:

dist_mtype = pd.DataFrame(data=0.0, index=traits_mtype.index, columns=id_to_layer_name.keys())


# In[23]:

from voxcell.traits import SpatialDistribution


# In[24]:

for layer in density:
    for mtype in density[layer]:
        d = traits_mtype[traits_mtype.mtype == mtype]
        dist_mtype.loc[d.index, layer_name_to_id[layer]] = density[layer][mtype]
dist_mtype /= dist_mtype.sum()
dist_mtype


# ## spatial distribution

# In[25]:

sd = SpatialDistribution(annotation, dist_mtype, traits_mtype)


# In[26]:

chosen_mtype = sd.assign(new_cells.positions)


# In[27]:

property_mtype = sd.collect_traits(chosen_mtype, ['mtype'])


# In[28]:

new_cells.add_properties(property_mtype)


# performs intrinsic validations based on the densities

# In[29]:

def sum_cells_mtype(cells, mtype):
    ''' count the number of cells of a given mtype'''
    return np.count_nonzero(cells.properties[cells.properties.mtype == mtype].mtype)


def get_density_delta(ref_densities, cells, layer_volumes):
    ''' get a panda dataframe for the density delta
    between ref_densities and given cell collection
    '''
    density_delta = pd.DataFrame(columns=['delta %'])
    for layer_name in ref_densities:
        for mtype in ref_densities[layer_name]:
            ref_density = ref_densities[layer_name][mtype]
            nb_cells = sum_cells_mtype(cells, mtype)
            model_density = float(nb_cells) / layer_volumes[layer_name]
            ratio = ((model_density / ref_density) - 1.0) * 100
            density_delta.loc[mtype] = round(ratio, 2)
    return density_delta

get_density_delta(density, new_cells, layer_volumes)


# In[30]:

VoxcellWidget().show_property('mtype', new_cells, display_parameters={"particle_size": 1.5})


# # morphology assignment

# for each mtype, morphology models are uniformly distributed.

# morphology_models lists are built based on a sample from https://collab.humanbrainproject.eu/#/collab/375/nav/3410
# please update it with the morphologies you want to take into account.

# In[31]:

morphology_models = {
    "BC": ["189-1-15dw.CNG", "189-1-3dw.CNG", "189-1-5dw.CNG", "189-1-9dw.CNG"],
    "GrC": ["210710C0.CNG", "240710C0.CNG", "270111C0.CNG", "270111C3.CNG", "Golgi-cell-051108-C0-cell1.CNG"],
    "PC": ["Purkinje-slice-ageP35-1.CNG", "Purkinje-slice-ageP35-2.CNG"],
    "SC": ["189-1-10dw.CNG", "189-1-12dw.CNG", "189-1-16dw.CNG", "189-1-1dw.CNG", "189-1-27dw.CNG", "189-1-6dw.CNG", "189-1-7dw.CNG"],
    "Glomeruli": ["Glomeruli-example"],
    "GoC": ["GoC-Example"]
}


# ## traits

# In[32]:

traits_morph = pd.DataFrame([[mtype, morph] for layer in density
                             for mtype in density[layer]
                             for morph in morphology_models[mtype]],
                            columns=["mtype", "morph_name"])


# In[33]:

traits_morph


# ## distribution

# In[34]:

dist_morphology = pd.DataFrame(data=0.0, index=traits_morph.index, columns=id_to_layer_name.keys())


# In[35]:

from voxcell.traits import SpatialDistribution


# In[36]:

for layer in density:
    for mtype in density[layer]:
        for morph_name in morphology_models[mtype]:
            d = traits_morph[(traits_morph.mtype == mtype) & (traits_morph.morph_name == morph_name)]
            dist_morphology.loc[d.index, layer_name_to_id[layer]] = 1
dist_morphology /= dist_morphology.sum()
dist_morphology


# ## spatial distribution

# In[37]:

sd = SpatialDistribution(annotation, dist_morphology, traits_morph)


# In[38]:

chosen_morph = sd.assign_conditional(new_cells.positions, property_mtype)


# In[39]:

property_morph = sd.collect_traits(chosen_morph, ['morph_name'])


# In[40]:

new_cells.add_properties(property_morph)


# perform instrinsic validations based on densities

# In[41]:

get_density_delta(density, new_cells, layer_volumes)


# In[42]:

VoxcellWidget().show_property('morph_name', new_cells, display_parameters={"particle_size": 1.5})


# # orientation assignment (WIP)

# In[43]:

from voxcell import vector_fields as vf


# In[44]:

v_right = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                        np.array([1, 0, 0]))


# In[45]:

v_up = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                     np.array([0, 1, 0]))


# In[46]:

v_fwd = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                      np.array([0, 0, 1]))


# In[47]:

fields = vf.combine_vector_fields([v_right, v_up, v_fwd])


# In[48]:

orientation_field = VoxelData(fields, annotation.voxel_dimensions, annotation.offset)


# In[49]:

orientation_field.raw.shape


# In[50]:

sub_fields = vf.split_orientation_field(orientation_field.raw)
VoxcellWidget().show_vectors('Z', sub_fields[2], 5000, orientation_field.voxel_dimensions)


# In[51]:

VoxcellWidget().show_vectors('Y', sub_fields[1], 5000, orientation_field.voxel_dimensions)


# In[52]:

VoxcellWidget().show_vectors('X', sub_fields[0], 5000, orientation_field.voxel_dimensions)


# In[ ]:



