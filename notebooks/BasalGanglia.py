
# coding: utf-8

# In[1]:

from collections import namedtuple
import numpy as np
import pandas as pd
from voxcellview.widget import VoxcellWidget
from voxcell.core import CellCollection, VoxelData
from voxcell import build

get_ipython().magic(u'matplotlib inline')


# network properties from: https://collab.humanbrainproject.eu/#/collab/376/nav/3621

# units for length (heights, widths,...) are all in µm.

# # density

# ## layered annotations

# In[2]:

LayerDefinition = namedtuple("LayerDefinition", "name, height, id")
layer_definitions = [
    # unique layer
    LayerDefinition("unique", 500, 1),
]

volume_length = 500
volume_width = 500


# In[3]:

layer_ids = [l.id for l in reversed(layer_definitions)]
layer_heights = [l.height for l in reversed(layer_definitions)]


# In[4]:

voxel_side = 5.0 # arbitrary resolution. Lower values will use more memory.
layer_heights_voxel = np.round(np.array(layer_heights) / voxel_side).astype(np.uint)
volume_length_voxel = volume_length / voxel_side
volume_width_voxel = volume_width / voxel_side


# In[5]:

annotation = VoxelData(build.layered_annotation((volume_length_voxel, volume_width_voxel),
                                                layer_heights_voxel, layer_ids),
                       [voxel_side] * 3)


# In[6]:

VoxcellWidget().show_volume('annotation', annotation, display_parameters={"particle_size": 3})


# ## layer densities

# In[7]:

# unit is cells per µ^3
# original data are cells per mm^3
DENSITY_RATIO = 1e-9  # to convert between mm^3 and micron^3

density = {
    "unique": {
        "D1_MSN": 35e3 * DENSITY_RATIO,
        # Golgi Cells
        "D2_MSN": 35e3 * DENSITY_RATIO,
        # Granule Cells
        "FS": 700 * DENSITY_RATIO,
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
    layer_name = id_to_layer_name[l_id]
    voxel_raw[mask] = float(sum(density[id_to_layer_name[l_id]].values()))

voxel_density = VoxelData(voxel_raw, annotation.voxel_dimensions, annotation.offset)


# In[12]:

VoxcellWidget().show_volume('density', voxel_density, display_parameters={"particle_size": 3})


# # cell positions

# In[13]:

new_cells = CellCollection()


# In[14]:

layer_volumes = dict((layer.name, layer.height * volume_length * volume_width)
                     for layer in layer_definitions)


# In[15]:

total_cell_count = round(sum(layer_volumes[layer] * d
                             for layer in layer_volumes
                             for d in density[layer].values()))


# In[16]:

total_cell_count


# In[17]:

from brainbuilder.cell_positioning import cell_positioning


# In[18]:

new_cells.positions = cell_positioning(voxel_density, int(total_cell_count))


# In[19]:

VoxcellWidget().show_points('position', new_cells, display_parameters={"particle_size": 1})


# # mtype assignment

# ## traits

# In[20]:

traits_mtype = pd.DataFrame([[mtype] for layer in density
                             for mtype in density[layer]],
                            columns=["mtype"])


# ## distribution

# In[21]:

dist_mtype = pd.DataFrame(data=0.0, index=traits_mtype.index, columns=id_to_layer_name.keys())


# In[22]:

from voxcell.traits import SpatialDistribution


# In[23]:

for layer in density:
    for mtype in density[layer]:
        d = traits_mtype[traits_mtype.mtype == mtype]
        dist_mtype.loc[d.index, layer_name_to_id[layer]] = density[layer][mtype]
dist_mtype /= dist_mtype.sum()
dist_mtype


# ## spatial distribution

# In[24]:

sd = SpatialDistribution(annotation, dist_mtype, traits_mtype)


# In[25]:

chosen_mtype = sd.assign(new_cells.positions)


# In[26]:

property_mtype = sd.collect_traits(chosen_mtype, ['mtype'])


# In[27]:

new_cells.add_properties(property_mtype)


# performs intrinsic validations based on the densities

# In[28]:

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


# In[29]:

VoxcellWidget().show_property('mtype', new_cells, display_parameters={"particle_size": 1.5})


# # morphology assignment

# for each mtype, morphology models are uniformly distributed.

# need examples to be provided in https://collab.humanbrainproject.eu/#/collab/376/nav/3415
# please update it when the reconstructed morphologies will be available.

# In[30]:

morphology_models = {
    "D1_MSN": ["d1_example1", "d1_example2"],
    "D2_MSN": ["d2_example1", "d2_example2"],
    "FS": ["fs_example1", "fs_example2"],
}


# ## traits

# In[31]:

traits_morph = pd.DataFrame([[mtype, morph] for layer in density
                             for mtype in density[layer]
                             for morph in morphology_models[mtype]],
                            columns=["mtype", "morph_name"])


# In[32]:

traits_morph


# ## distribution

# In[33]:

dist_morphology = pd.DataFrame(data=0.0, index=traits_morph.index, columns=id_to_layer_name.keys())


# In[34]:

from voxcell.traits import SpatialDistribution


# In[35]:

for layer in density:
    for mtype in density[layer]:
        for morph_name in morphology_models[mtype]:
            d = traits_morph[(traits_morph.mtype == mtype) & (traits_morph.morph_name == morph_name)]
            dist_morphology.loc[d.index, layer_name_to_id[layer]] = 1
dist_morphology /= dist_morphology.sum()
dist_morphology


# ## spatial distribution

# In[36]:

sd = SpatialDistribution(annotation, dist_morphology, traits_morph)


# In[37]:

chosen_morph = sd.assign_conditional(new_cells.positions, property_mtype)


# In[38]:

property_morph = sd.collect_traits(chosen_morph, ['morph_name'])


# In[39]:

new_cells.add_properties(property_morph)


# perform instrinsic validations based on densities

# In[40]:

get_density_delta(density, new_cells, layer_volumes)


# In[41]:

VoxcellWidget().show_property('morph_name', new_cells, display_parameters={"particle_size": 1.5})


# # orientation assignment (WIP)

# In[42]:

from voxcell import vector_fields as vf


# In[43]:

v_right = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                        np.array([1, 0, 0]))


# In[44]:

v_up = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                     np.array([0, 1, 0]))


# In[45]:

v_fwd = vf.generate_homogeneous_field(np.ones(annotation.raw.shape, dtype=np.bool),
                                      np.array([0, 0, 1]))


# In[46]:

fields = vf.combine_vector_fields([v_right, v_up, v_fwd])


# In[47]:

orientation_field = VoxelData(fields, annotation.voxel_dimensions, annotation.offset)


# In[48]:

orientation_field.raw.shape


# In[49]:

sub_fields = vf.split_orientation_field(orientation_field.raw)
VoxcellWidget().show_vectors('Z', sub_fields[2], 5000, orientation_field.voxel_dimensions)


# In[50]:

VoxcellWidget().show_vectors('Y', sub_fields[1], 5000, orientation_field.voxel_dimensions)


# In[51]:

VoxcellWidget().show_vectors('X', sub_fields[0], 5000, orientation_field.voxel_dimensions)


# In[ ]:



