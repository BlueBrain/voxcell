Data Examples
~~~~~~~~~~~~~

The AIBS_ provides volumetric data for cell densities and gene markers free
to download from their website. The format is often uchar (8bit) grayscale values volume taken from
the reconstructed brain at 25 µm resolution.

- Download `general cell body position <http://api.brain-map.org/api/v2/well_known_file_download/113567585>`_
  (from `MouseBrain API`_ search for "atlasVolume:").  These are Nissl stained data.

- Download `glia body positions <http://mouse.brain-map.org/search/show?page_num=0&page_size=26&no_paging=false&exact_match=false&search_term=gfap&search_type=gene>`_.
  These are the gene expression stains of Gfap, so we have density/energy/intensity voxel information
  at lower resolution.

- Download `inhibitory/excitatory body positions <http://mouse.brain-map.org/search/show?page_num=0&page_size=26&no_paging=false&exact_match=false&search_term=gad&search_type=gene>`_
  (from `MouseBrain API`_
  search for 'Packing of 3-D volumetric data into a 1-D numerical array').
  These are the gene expression stains of Gad.

- Download the `atlas <http://api.brain-map.org/api/v2/well_known_file_download/197642854>`_
  (from `MouseBrain API`_, search for "annotation:").
  An atlas is a dataset where each voxel maps to a brain region.
  The format is in uint (32bit) of structural annotation volume at 25 µm resolution.
  The value represents the ID of the finest level structure annotated for the voxel.

Hierarchy
---------

The Hierarchy object represents the relationship bewteen the regions that compose an atlas.
Note that the 3-D mask for any structure is composed of all voxels of the atlas annotated for that
structure and for any of its descendants in the structure hierarchy.


Voxcell supports loading JSON files as support for hierarchy data:

.. code-block:: python

    from voxcell.nexus.voxelbrain import Atlas
    hierarchy = Atlas.open('/path/to/atlas').load_region_map()

Example
~~~~~~~

The AIBS_ provides a hierarchy free to download from their website.

Download the `structured hierarchy <http://api.brain-map.org/api/v2/structure_graph_download/1.json>`_.
Structures are organized in a hierarchy where each Structure has one parent
denoting a "part-of" relationship.


.. _AIBS: http://alleninstitute.org/
.. _`MouseBrain API`: http://help.brain-map.org//display/mousebrain/API
