''' a ipython notebook extension to display voxcell data '''
import base64
import numpy as np
from ipywidgets import Widget # pylint: disable=F0401
from traitlets import (Unicode, Int, List, Dict) # pylint: disable=F0401
from IPython.display import display # pylint: disable=F0401
from voxcellview import viewer # pylint: disable=F0401
from voxcell import core


class VoxcellWidget(Widget): # pylint: disable=R0901
    ''' display voxcell data '''
    _view_module = Unicode('nbextensions/voxcellview/voxcellview/voxcellview', sync=True)
    _view_name = Unicode('CircuitView', sync=True)
    _model_module = Unicode('nbextensions/voxcellview/voxcellview/voxcellview', sync=True)
    _model_name = Unicode('CircuitModel', sync=True)

    # TODO: investigate why bytes are not passed properly.
    #bytes_data = Bytes(b'', sync=True)
    bytes_data = Unicode('', sync=True)
    shape = List(Int, sync=True)
    name = Unicode('', sync=True)
    dtype = Unicode('', sync=True)

    display_parameters = Dict({}, sync=True)

    def _show(self, block, display_parameters=None):
        ''' show numpy binary data'''
        self.bytes_data = base64.b64encode(block.tobytes())
        if display_parameters is None:
            display_parameters = {}
        self.display_parameters = display_parameters
        display(self)

    def show_points(self, name, cells, display_parameters=None):
        ''' display a bunch of positions '''
        self.show_property(name, cells, '.pts', display_parameters)

    # TODO: remove the remaining extension switch in js code
    def show_property(self, name, cells, extension='.pts', display_parameters=None):
        ''' display a bunch of positions with properties '''
        block = viewer.serialize_points(cells, name)
        self.name = name + extension
        self._show(block, display_parameters)

    def show_vectors(self, name, field, point_count, voxel_dimensions, display_parameters=None):
        ''' display a vector field '''
        block = viewer.serialize_vector_field(field, point_count, voxel_dimensions)
        self.name = name + '.vcf'
        self._show(block, display_parameters)

    def show_volume(self, name, voxel, display_parameters=None):
        ''' display VoxelData '''
        if isinstance(voxel, np.ndarray):
            if voxel.dtype == np.bool:
                voxel = voxel * np.ones_like(voxel, dtype=np.uint8)
            voxel = core.VoxelData(voxel, (1, 1, 1))

        block = voxel.raw.transpose()
        self.name = name + '.raw'
        self.shape = voxel.raw.shape
        self.dtype = str(voxel.raw.dtype)
        self._show(block, display_parameters)
