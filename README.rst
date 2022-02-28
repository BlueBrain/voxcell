Overview
========

Libraries for:

* handling "traits fields" and collections and the logic to assign them.
* volumetric data within NRRD files
* Cell collection access / writer.
* to build, transform and handle fields of vectors and orientations.
* querying `Region hierarchy tree`, such as the one available from the `Allen Institute for Brain Science (AIBS)`_: `Mouse Brain Atlas Ontology's StructureGraph`_ (aka 1.json).

Installation
============

.. code-block:: bash

    git clone https://github.com/BlueBrain/voxcell
    cd voxcell
    pip install -e .


Acknowledgements
================

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see LICENSE.txt and AUTHORS.txt respectively.

Copyright © 2022 Blue Brain Project/EPFL

.. _`Allen Institute for Brain Science (AIBS)`: https://alleninstitute.org/what-we-do/brain-science/
.. _`Mouse Brain Atlas Ontology's StructureGraph`: http://api.brain-map.org/api/v2/structure_graph_download/1.json
