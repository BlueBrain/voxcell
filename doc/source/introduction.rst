Introduction
============

Installation
------------

Voxcell is pip installable. It defines its dependencies explicitly so that they are pulled in automatically.

.. code-block:: bash

   $ virtualenv venv
   $ source venv/bin/activate
   $ pip install --index-url http://bbpgb019.epfl.ch:9090/simple/ --pre voxcell


Quick Example
-------------

The following snippet builds a spatial distribution for a variable "type" for the entire brain
but taking into account that type A is much more probable in Isocortex than anywhere else.

.. code-block:: python

    from voxcell.core import VoxelData, Hierarchy, CellCollection
    from voxcell.traits import SpatialDistribution
    from voxcell import build

    import numpy as np
    import pandas as pd

    atlas = VoxelData.load_metaio('data/P56_Mouse_annotation.mhd')
    hierarchy = Hierarchy.load('data/annotation_hierarchy.json')

    isocortex_mask = build.mask_by_region_names(atlas.raw, hierarchy, ['Isocortex'])

    field = np.zeros_like(atlas.raw)
    field[isocortex_mask] = 1
    distributions = pd.DataFrame({0: [0.5, 0.5], 1: [0.95, 0.05]})
    traits = pd.DataFrame({'type': ['A', 'B']})

    sd = SpatialDistribution(field, distributions, traits)



