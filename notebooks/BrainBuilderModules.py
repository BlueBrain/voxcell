
# coding: utf-8

# ### viewers setup

# In[1]:

from voxcellview.widget import VoxcellWidget


# ### loaders initialization

# In[2]:

from voxcell.core import CellCollection, VoxelData, Hierarchy
from brainbuilder.utils import bbp

cells = CellCollection()


# # Brain Builder modules

# ##   Build.Region: Region of Interest

# In[3]:

from brainbuilder.select_region import select_region, select_hemisphere


# ### input parameters

# In[4]:

annotation = VoxelData.load_metaio("../data/P56_Mouse_annotation/annotation.mhd")
hierarchy = Hierarchy.load("../data/P56_Mouse_annotation/annotation_hierarchy.json")
full_density = VoxelData.load_metaio("../data/atlasVolume/atlasVolume.mhd")
region_name = "Primary somatosensory area"


# ### run module

# In[5]:

density = select_region(annotation.raw, full_density, hierarchy, region_name)
density.raw = select_hemisphere(density.raw)


# ### output

# In[6]:

VoxcellWidget().show_volume('density', density)


# ##  Build.Cells:  Cell Positions

# In[7]:

from brainbuilder.cell_positioning import cell_positioning


# ### input parameters

# In[8]:

total_cell_count = 200000


# ### run module

# In[9]:

cells.positions = cell_positioning(density, total_cell_count)


# ### output

# In[10]:

VoxcellWidget().show_points('position', cells)


# ## Build.EI:  E-I ratios

# In[11]:

from brainbuilder.assignment_synapse_class import assign_synapse_class_from_spatial_dist


# ### input parameters

# In[12]:

recipe_filename = "../data/bbp_recipe/builderRecipeAllPathways.xml"
recipe_sdist = bbp.load_recipe_as_spatial_distribution(recipe_filename, annotation, hierarchy, region_name)


# ### run module

# In[13]:

chosen_synapse_class = assign_synapse_class_from_spatial_dist(cells.positions, recipe_sdist)
cells.add_properties(chosen_synapse_class)


# ### output

# In[14]:

VoxcellWidget().show_property('synapse_class', cells)


# ## Build.Composition.ME: METype for Soma

# In[15]:

from brainbuilder.assignment_metype import assign_metype


# ### run module

# In[16]:

chosen_me = assign_metype(cells.positions, cells.properties.synapse_class, recipe_sdist)
cells.add_properties(chosen_me)


# ### output

# #### mtypes

# In[17]:

VoxcellWidget().show_property('mtype', cells)


# #### etypes

# In[18]:

VoxcellWidget().show_property('etype', cells)


# ## Build.Placement: Morphology assignment

# In[19]:

from brainbuilder.assignment_morphology import assign_morphology


# ### input parameters

# In[20]:

neurondb_filename = "../data/bbp_recipe/neurondb.dat"
neuron_sdist = bbp.load_neurondb_v4_as_spatial_distribution(neurondb_filename, annotation, hierarchy, region_name, percentile=0.92)


# ### run module

# In[21]:

chosen_morphology = assign_morphology(cells.positions, cells.properties[['mtype', 'etype']], neuron_sdist)
cells.add_properties(chosen_morphology)


# ### output

# In[22]:

VoxcellWidget().show_property('morphology', cells)


# ## Orientation assignement

# In[23]:

from brainbuilder.orientation_field_sscx import compute_orientation_field
from brainbuilder.assignment_orientation import assign_orientations


# ### run module

# In[24]:

orientation_field = compute_orientation_field(annotation, hierarchy, region_name)
cells.orientations = assign_orientations(cells.positions, orientation_field)


# ### output

# In[25]:

#notebook.show_placement('placement', cells)

