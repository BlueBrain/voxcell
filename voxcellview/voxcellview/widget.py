''' a ipython notebook extension to display voxcell data '''
import base64
import os
import tempfile
import shutil
import numpy as np
import pylru
from tqdm import tqdm
from neurom.io import load_data
import ipywidgets as widgets
from traitlets import (Unicode, Int, List, Dict, Bytes) # pylint: disable=F0401
from IPython.display import display, HTML # pylint: disable=F0401
from voxcellview import viewer # pylint: disable=F0401


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

    spikes = List().tag(sync=True)

    # supported d-types of the JS client BrainBuilderViewer -> buildRaw -> DTYPE_TYPE_MAP
    SUPPORTED_DTYPES = ['uint8', 'uint16', 'uint32', 'float32']

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

    def show_volume(self, name, data, display_parameters=None):
        ''' display numpy voxel data '''
        if data.dtype == np.bool:
            data = data * np.ones_like(data, dtype=np.uint8)
        self.name = name + '.raw'
        self.shape = data.shape
        self.dtype = str(data.dtype)

        if self.dtype not in self.SUPPORTED_DTYPES:
            return display(HTML('<div style="color:red">D-Type %s is not supported' % self.dtype))

        block = data.transpose()
        self._show(block=block, display_parameters=display_parameters)

    def show_morphologies(self, cells, loader_fct):
        ''' display circuit with morphologies
        Args:
           cells: a pandas DataFrame with ['x', 'y', 'z', 'orientation', 'morphology'] columns
           loader_fct: a function that takes a morphology_name and returns the morphology as base64
        '''
        self._show()
        for _, item in tqdm(cells.iterrows()):
            self.send({
                'position': serialize_floats(item[['x', 'y', 'z']].values),
                'orientation': serialize_floats(item['orientation'].values),
                'data': loader_fct(item['morphology']),
            })

    def show_spikes(self, name, cells, spks, display_parameters=None):
        '''
        Display spikes

        Args:
            spks: a pandas DataFrame with ['time', 'id'] columns.
                  It represents time and the id of the spiking cell.
                  Can be loaded with something like the following:
                      pd.read_csv("out.dat",
                                  header=None,
                                  names=['time', 'id'],
                                  dtype={'time': np.float64, 'id': np.int32},
                                  skiprows=1,
                                  delim_whitespace=True)
        '''
        self.show_property(name, cells, '.pts', display_parameters)
        self.spikes = [group.id.tolist() for _, group in spks.groupby('time')]


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
            self._get = pylru.FunctionCacheManager(
                self._get_as_b64, size=loader_cache_size
            )
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
