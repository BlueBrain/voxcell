''' a ipython notebook extension to display voxcell data '''
import base64
import random
import os
import tempfile
import shutil
import numpy as np
import functools32
from tqdm import tqdm
from neurom.io import load_data
import ipywidgets as widgets
from traitlets import (Unicode, Int, List, Dict, Bytes) # pylint: disable=F0401
from IPython.display import display # pylint: disable=F0401
from voxcellview import viewer # pylint: disable=F0401
from voxcell import core

# TODO: what is that for?
@widgets.register('voxcellview.Viewer')
class VoxcellWidget(widgets.DOMWidget): # pylint: disable=R0901
    ''' display voxcell data '''
    _view_module = Unicode('voxcellview').tag(sync=True)
    _view_name = Unicode('CircuitView').tag(sync=True)
    _model_module = Unicode('voxcellview').tag(sync=True)
    _model_name = Unicode('CircuitModel').tag(sync=True)

    bytes_data = Bytes(b'').tag(sync=True)
    shape = List(Int).tag(sync=True)
    name = Unicode('').tag(sync=True)
    dtype = Unicode('').tag(sync=True)

    display_parameters = Dict({}).tag(sync=True)

    def _show(self, block=None, display_parameters=None):
        ''' show numpy binary data'''
        if block is not None:
            self.bytes_data = block.tobytes()
        if display_parameters is None:
            display_parameters = {}
        self.display_parameters = display_parameters
        display(self)

    def show_points(self, name, cells, display_parameters=None):
        ''' display a bunch of positions '''
        self.show_property(name, cells, '.pts', display_parameters)

    # TODO: remove the remaining extension switch in js code
    def show_property(self, name, cells, extension='.pts', display_parameters=None, color_map=None):
        ''' display a bunch of positions with properties '''
        block = viewer.serialize_points(cells, name, color_map)
        self.name = name + extension
        self._show(block=block, display_parameters=display_parameters)

    def show_vectors(self, name, field, point_count, voxel_dimensions, display_parameters=None):
        ''' display a vector field '''
        block = viewer.serialize_vector_field(field, point_count, voxel_dimensions)
        self.name = name + '.vcf'
        self._show(block=block, display_parameters=display_parameters)

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
        self._show(block=block, display_parameters=display_parameters)

    def show_morphologies(self, cells,
                          attribute,
                          loader_fct,
                          display_probability=1.0):
        ''' display circuit with morphologies
        Args:
           cells: a CellCollection
           attribute: the attribute holding the morphology name in the cells.properties
           loader_fct: a function that takes a morphology_name and returns the morphology as base64
           display_probability: the change of a morphology to be displayed (between 0 and 1.0)
        '''
        self._show()

        displayed_cells = np.nonzero(np.random.random(len(cells.positions)) < display_probability)[0]
        for k in tqdm(displayed_cells):
            neuron_data = loader_fct(cells.properties[attribute][k])
            position = serialize_floats(cells.positions[k])
            orientation = serialize_floats(cells.orientations[k])

            self.send({'position': position,
                       'orientation': orientation,
                       'data': neuron_data})


def serialize_floats(numpy_array):
    ''' convert a numpy array of floats to base64 '''
    f_array = numpy_array.astype(np.float32)
    return base64.b64encode(f_array.tobytes())


class MorphologyLoader(object):
    ''' load morphology files as base64 from local or remote locations '''
    def __init__(self, morphology_dir, file_extension='.swc',
                 document_client=None, loader_cache_size=None):
        '''
        Args:
        morphology_dir: a directory containing the morphologies (remote or local)
        file_extension: a suffix to load the morphology
        document_client: a DocumentClient object if morphologies are remote
        loader_cache_size: size of the LRU cache. None being no cache.
        '''
        self.morphology_dir = morphology_dir
        self.file_extension = file_extension
        self.document_client = document_client
        self.loader_cache_size = loader_cache_size
        self.tmp_dir = None
        if loader_cache_size is not None:
            self._get = (functools32.lru_cache(maxsize=loader_cache_size)
                         (self._get_as_b64))
        else:
            self._get = self._get_as_b64

    def __enter__(self):
        if self.document_client:
            self.tmp_dir = tempfile.mkdtemp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir)

    def get(self, morph_name):
        ''' load a morphology with neurom and return it as base 64
        through document_client if provided and from local file system otherwise
        '''
        return self._get(morph_name)

    def _get_as_b64(self, morph_name):
        ''' load the morphology with neurom and returns it as base64 '''
        file_name = morph_name + self.file_extension
        morphology_path = os.path.join(self.morphology_dir, file_name)
        if self.document_client:
            tmp_path = os.path.join(self.tmp_dir, file_name)
            self.document_client.download_file(morphology_path,
                                               tmp_path)
            path_to_load = tmp_path
        else:
            path_to_load = morphology_path

        data = load_data(path_to_load)
        return serialize_floats(data.data_block)
