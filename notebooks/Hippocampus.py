
# coding: utf-8

# In[1]:

from voxcell.core import CellCollection, VoxelData, Hierarchy
from voxcell import build, math
from voxcellview.widget import VoxcellWidget
from brainbuilder.utils import bbp
import numpy as np

cells = CellCollection()


# # Brain Builder modules

# ## Region Select

# In[2]:

from brainbuilder.select_region import select_hemisphere


# In[3]:

hierarchy = Hierarchy.load("../data/P56_Mouse_annotation/annotation_hierarchy.json")


# In[4]:

region_name = hierarchy.find('acronym', 'CA1')[0].data['name']
print 'Building:', region_name


# In[5]:

annotation = VoxelData.load_metaio("../data/P56_Mouse_annotation/annotation.mhd")

hippo_mask = build.mask_by_region_ids(annotation.raw, hierarchy.collect('name', region_name, 'id'))
aabb = math.minimum_aabb(hippo_mask)

hippo_mask = math.clip(hippo_mask, aabb)
annotation = annotation.clipped(aabb)


# In[6]:

region_layers_map = {
    'hippocampal fissure': 1,
    'Field CA1, stratum lacunosum-moleculare': 2,
    'Field CA1, stratum radiatum': 3,
    'Field CA1, pyramidal layer': 4,
    'Field CA1, stratum oriens': 5,
    'alveus': 6
}

region_layers_map = dict((hierarchy.find('name', name)[0].data['id'], (layerid,))
                         for name, layerid in region_layers_map.iteritems())


# In[7]:

recipe_filename = "../data/hippo_recipe/builderRecipeAllPathways.xml"

density = bbp.load_recipe_density(recipe_filename, annotation, region_layers_map)

density.raw = select_hemisphere(density.raw)


# In[8]:

VoxcellWidget().show_volume('density', density)


# ## Positions

# In[9]:

from brainbuilder.cell_positioning import cell_positioning


# In[10]:

total_cell_count = 350000


# In[11]:

cells.positions = cell_positioning(density, total_cell_count)


# In[12]:

from voxcellview.widget import VoxcellWidget


# In[13]:

widget = VoxcellWidget()


# In[14]:

VoxcellWidget().show_points('position', cells)


# ## widget.show_points('position', cells)

# ## Build.EI:  E-I ratios

# In[15]:

from brainbuilder.assignment_synapse_class import assign_synapse_class_from_spatial_dist


# ### input parameters

# In[16]:

recipe_filename = "../data/hippo_recipe/builderRecipeAllPathways.xml"
recipe_data = bbp.get_distribution_from_recipe(recipe_filename)


# In[17]:

recipe_sdist = bbp.transform_recipe_into_spatial_distribution(annotation, recipe_data, region_layers_map)


# ### run module

# In[18]:

chosen_synapse_class = assign_synapse_class_from_spatial_dist(cells.positions, recipe_sdist)
cells.add_properties(chosen_synapse_class)


# In[19]:

VoxcellWidget().show_property('synapse_class', cells)


# In[20]:

import numpy as np
vals, nums = np.unique(cells.properties.synapse_class, return_counts=True)
print '\n'.join('%s  total: %d  percentage: %.2f%%' % (n, t, p * 100)
                for n, t, p in zip(vals, nums, nums.astype(np.float) / total_cell_count))


# ## Build.Composition.ME: METype for Soma

# In[21]:

from brainbuilder.assignment_metype import assign_metype


# #### mtypes

# In[22]:

chosen_me = assign_metype(cells.positions, cells.properties.synapse_class, recipe_sdist)
cells.add_properties(chosen_me)


# #### mtypes

# In[23]:

VoxcellWidget().show_property('mtype', cells)


# #### etypes

# In[24]:

VoxcellWidget().show_property('etype', cells)


# ## Build.Placement: Morphology assignment

# In[25]:

from brainbuilder.assignment_morphology import assign_morphology
from scipy.ndimage import distance_transform_edt


# ### input parameters

# In[26]:

neurondb_filename = "../data/hippo_recipe/v4neuronDB.dat"
neurondb = bbp.load_neurondb_v4(neurondb_filename)

# "outside"  is tagged in the annotation_raw with 0
# This will calculate, for every voxel, the euclidean distance to
# the nearest voxel tagged as "outside" the brain
# TODO use something else for hippocampus
distance_to_pia = distance_transform_edt(hippo_mask)


# In[27]:

neuron_sdist = bbp.transform_neurondb_into_spatial_distribution(annotation,
                                                                 neurondb,
                                                                 region_layers_map,
                                                                 distance_to_pia,
                                                                 percentile=0.92)


# ### run module

# In[28]:

chosen_morphology = assign_morphology(cells.positions, cells.properties[['mtype', 'etype']], neuron_sdist)
cells.add_properties(chosen_morphology)


# ### output

# In[29]:

VoxcellWidget().show_property('morphology', cells)


# ## Orientation assignment

# ### input

# In[30]:

from brainbuilder.orientation_field_hippo import compute_orientation_field

orientation_field = compute_orientation_field(annotation, hierarchy, 'Field CA1')


# In[31]:

from voxcell import vector_fields as vf

sub_fields = vf.split_orientation_field(orientation_field.raw)
VoxcellWidget().show_vectors('Z', select_hemisphere(sub_fields[2]), 5000, orientation_field.voxel_dimensions)


# In[32]:

VoxcellWidget().show_vectors('Y', select_hemisphere(sub_fields[1]), 5000, orientation_field.voxel_dimensions)


# In[33]:

VoxcellWidget().show_vectors('X', select_hemisphere(sub_fields[0]), 5000, orientation_field.voxel_dimensions)


# ### run module

# In[34]:

from brainbuilder.assignment_orientation import assign_orientations

cells.orientations = assign_orientations(cells.positions, orientation_field)


# ### output

# In[35]:

# notebook.show_placement('placement', cells)

