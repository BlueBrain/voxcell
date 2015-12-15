Data Objects
============

VoxelData
---------

The central container class in voxcell. Uses a 3-rank numpy array to represent a series of voxels
in space and some metadata to identify the volume covered by the data relative to a global atlas.
The meaning of the value contained by each voxel will change depending on the usage.

Voxcell supports loading and saving MetaIO_ files as support for basic voxel data:

.. code-block:: python

    from voxcell.core import VoxelData
    density = VoxelData.load_metaio('/path/to/file.mhd')
    density.save_metaio('/path/to/file.mhd')

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


Voxcell supports loading JSON_ files as support for hierarchy data:

.. code-block:: python

    from voxcell.core import Hierarchy
    hierarchy = Hierarchy.load('/path/to/file.json')
    hierarchy.save_metaio('/path/to/file.mhd')

Example
~~~~~~~

The AIBS_ provides a hierarchy free to download from their website.

Download the `structured hierarchy <http://api.brain-map.org/api/v2/structure_graph_download/1.json>`_.
Structures are organized in a hierarchy where each Structure has one parent
denoting a "part-of" relationship.

CellCollection
--------------

A container for a collection of cells that are under construction. It uses numpy arrays to store
multi-dimensional numerical attributes (3-D position and orientation) and a pandas.DataFrame object
to represent any other property.

Voxcell supports saving cell collections as :doc:`mvd3` files:

.. code-block:: python

    from voxcell.core import CellCollection
    cells = CellCollection()
    # ...
    cells.save('/path/to/file.h5')


.. _MetaIO: http://www.itk.org/Wiki/MetaIO/Documentation
.. _JSON: http://www.json.org
.. _AIBS: http://alleninstitute.org/
.. _`MouseBrain API`: http://help.brain-map.org//display/mousebrain/API


SpatialDistribution
-------------------

The heavy lifting class in voxcell. It represents a discrete random variable whose probability
distribution varies in 3-D space. You can also think of it as a collection of different probability
distributions, all of them for the same variable, that are associated with different regions of
space.

Contains a *field* VoxelData_ object to cover a volume of space, a *distributions* pandas.DataFrame
object to represent probability distributions and a *traits* pandas.DataFrame object to represent the
possible values of the variable. Each voxel from *field* contains the index of a
column of *distributions* (a single probability distribution). Each row of *distributions* matches
one row in *traits*.

Example
~~~~~~~
Imagine a SpatialDistribution object with the following values:

- field:

    [0, 0, 1, 1]

- distributions:

    +---+------+------+
    |   | 0    | 1    |
    +===+======+======+
    | 0 | 0.5  | 0.95 |
    +---+------+------+
    | 1 | 0.5  | 0.05 |
    +---+------+------+
    | 2 | 0.0  | 0.0  |
    +---+------+------+

- traits:

    +---+------+-------+
    |   | type | color |
    +===+======+=======+
    | 0 |   A  |  blue |
    +---+------+-------+
    | 1 |   B  |  red  |
    +---+------+-------+
    | 2 |   B  |  blue  |
    +---+------+-------+


In this case, the distribution for a random variable "type" can take two values: "A" or "B".
The field covers 4 voxels: the last two have a much higher probability of generating a value
of "A+blue" than "B+red", while the first two have the same probability for both options.
The option "B+blue" is just impossible in any case.

Note that since traits is a table the variable can be multivalue (in this case composed of the
variables type and color).