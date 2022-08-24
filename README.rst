Overview
========

This is a library to build circuits and atlases. It contains tools to handle

* "traits fields" and collections and the logic to assign them.
* volumetric data within NRRD files
* Cell collection access / writer.
* to build, transform and handle fields of vectors and orientations.
* querying `Region hierarchy tree`, such as the one available from the `Allen Institute for Brain Science (AIBS)`_: `Mouse Brain Atlas Ontology's StructureGraph`_ (aka 1.json).

Installation
============

Install from PyPI:

.. code-block:: bash

    pip install voxcell

Or an editable install:

.. code-block:: bash

    git clone https://github.com/BlueBrain/voxcell
    cd voxcell
    pip install -e .

Examples
========

To use the following examples, one must download an NRRD file and the Ontology Structure:

.. code-block:: bash

    curl -o brain_regions.nrrd http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_100.nrrd
    curl -o hierarchy.json http://api.brain-map.org/api/v2/structure_graph_download/1.json

One can open NRRD files, and perform operations on them:

.. code-block:: python

    import voxcell
    voxels = voxcell.VoxelData.load_nrrd('brain_regions.nrrd')
    print(voxels.voxel_dimensions)  # prints array([100., 100., 100.], dtype=float32)

One can also use the `Atlas` object to load at both the atlas and the hierarchy:

.. code-block:: python

    import numpy as np
    from voxcell.nexus.voxelbrain import Atlas
    atlas = Atlas.open('.')
    brain_regions = atlas.load_data('brain_regions')
    rm = atlas.load_region_map()
    # count the number of voxels in the VIS region, and all its descendents
    ids = rm.find('VIS', 'acronym', with_descendants=True)
    np.count_nonzero(np.isin(brain_regions.raw, list(ids)))

Citation
========

When you use this software, we kindly ask you to cite the following DOI:

.. image:: https://zenodo.org/badge/451807050.svg
   :target: https://zenodo.org/badge/latestdoi/451807050
   

Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see LICENSE.txt and AUTHORS.txt respectively.

Copyright © 2022 Blue Brain Project/EPFL

.. _`Allen Institute for Brain Science (AIBS)`: https://alleninstitute.org/what-we-do/brain-science/
.. _`Mouse Brain Atlas Ontology's StructureGraph`: http://api.brain-map.org/api/v2/structure_graph_download/1.json
