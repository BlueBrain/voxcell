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
    from voxcell.positions import create_cell_positions
    from voxcell import build

    import numpy as np
    import pandas as pd

    # Use an atlas to construct a spatial distribution of types A/B in a brain
    atlas = VoxelData.load_nrrd('data/P56_Mouse_annotation.nrrd')
    hierarchy = Hierarchy.load('data/annotation_hierarchy.json')

    isocortex_mask = build.mask_by_region_names(atlas.raw, hierarchy, ['Isocortex'])

    field = np.zeros_like(atlas.raw)
    field[isocortex_mask] = 1
    distributions = pd.DataFrame({ 1: [0.95, 0.05],  # isocortex
                                   0: [0.5, 0.5],})  # rest of the brain
    sdist = SpatialDistribution(field, distributions, pd.DataFrame({'type': ['A', 'B']}))

    # Use a density to build a collection of cells in space
    density = Hierarchy.load('data/annotation_hierarchy.json')

    cells = CellCollection()
    cells.positions = create_cell_positions(density, 1000)

    # Use the spatial distribution to assign a type A/B to each cell
    chosen = sdist.assign(cells.positions)
    cells.add_properties(sdist.collect_traits(chosen, ['type']))

