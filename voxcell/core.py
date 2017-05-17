'''core container classes'''
#pylint:disable=unused-import

from voxcell import deprecate, VoxcellError
from voxcell.cell_collection import CellCollection
from voxcell.hierarchy import RegionMap, Hierarchy
from voxcell.voxel_data import VoxelData


deprecate.warn("""
    voxcell.core is deprecated.
    Please change your imports as following:
        from voxcell.core import X -> from voxcell import X
""")
