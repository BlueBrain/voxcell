
Getting Started for Developers
------------------------------

1) Prepare the data directory: run download_data.sh



Ubuntu python modules (unsupported)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sudo apt get install python-h5py python-numpy python-scipy




File Formats
------------

An overview of the file formats used by the BrainBuilder.

Allen Institute Cell Body Voxel Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format:
MetaIO from Kitware (http://www.itk.org/Wiki/MetaIO/Documentation)
uchar (8bit) grayscale Nissl volume of the reconstructed brain at 25 µm resolution

For more details go to http://help.brain-map.org//display/mousebrain/API and
search for 'Packing of 3-D volumetric data into a 1-D numerical array'.

Examples:
For general cell body position:
http://api.brain-map.org/api/v2/well_known_file_download/113567585
From http://help.brain-map.org/display/mousebrain/API (search for "atlasVolume:")

For glia body positions.
These are the gene expression stains of Gfap, so we have density/energy/intensity voxel
information at lower resolution (for example):
http://mouse.brain-map.org/search/show?page_num=0&page_size=26&no_paging=false&exact_match=false&search_term=gfap&search_type=gene

For inhibitory/excitatory body positions.
These are the gene expression stains of Gad:
http://mouse.brain-map.org/search/show?page_num=0&page_size=26&no_paging=false&exact_match=false&search_term=gad&search_type=gene


Allen Institute Region Voxel Annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format:
Voxel mapping to brain region.
uint (32bit) structural annotation volume at 25 µm resolution.
The value represents the ID of the finest level structure annotated for the voxel.

Note: the 3-D mask for any structure is composed of all voxels annotated for that structure and
all of its descendants in the structure hierarchy.

Examples:
http://api.brain-map.org/api/v2/well_known_file_download/197642854
From http://help.brain-map.org/display/mousebrain/API (search for "annotation:")


Allen Institute Region Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format:
JSON
The structures are organized in a hierarchy where each Structure has one parent and
denotes a "part-of" relationship.

Example:
http://api.brain-map.org/api/v2/structure_graph_download/1.json
