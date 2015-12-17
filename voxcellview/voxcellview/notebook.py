'''simple integration of the brain builder viewer with ipython notebooks'''

import os
import shutil
from os.path import join as joinp
from IPython.display import HTML, display  # pylint: disable=F0401
import urllib

import numpy as np
from voxcell import core
from voxcellview import viewer


class NotebookViewer(object):
    '''encapsulates the integration of the webgl in an ipython notebook'''

    def __init__(self, tmp_path='out', server_path='../../static'):
        ''' NotebookViewer

        Args:
            tmp_path: the location where the temporary output will be stored relative to the
                web viewer source directory. To ensure that the server is able to load it,
                it must placed under the web viewer source directory.
            server_path: path to the server of the web viewer code.
        '''
        self.server_path = server_path

        self.output_rel_path = tmp_path
        self.output_abs_path = joinp(os.path.dirname(__file__), 'webviewer', self.output_rel_path)

        if not os.path.isdir(self.output_abs_path):
            os.makedirs(self.output_abs_path)

    def clean(self):
        '''delete the tmp folder'''
        shutil.rmtree(self.output_abs_path)

    def show(self, filename, display_parameters=None):
        '''display the given local file'''
        url_parameters = ""
        if display_parameters:
            url_parameters = "?" + urllib.urlencode(display_parameters)

        index_path = joinp(self.server_path, 'index.html')
        url = index_path + url_parameters + '#' + joinp(self.output_rel_path, filename)
        html = (
            '<iframe src="{url}"'
            ' scrolling="no" width="700" height="350" allowfullscreen></iframe>\n'
        ).format(url=url)

        display(HTML(html))

    def show_volume(self, name, voxel, display_parameters=None):
        '''save volumetric data locally and display it
        it may be a VoxelData object or a 3D numpy array'''

        if isinstance(voxel, np.ndarray):
            if voxel.dtype == np.bool:
                voxel = voxel * np.ones_like(voxel, dtype=np.uint8)
            voxel = core.VoxelData(voxel, (1, 1, 1))

        filename_mhd = name + '.mhd'
        voxel.save_metaio(joinp(self.output_abs_path, filename_mhd))
        self.show(filename_mhd, display_parameters)

    def show_points(self, name, cells, display_parameters=None):
        '''save a bunch of positions locally and display them'''
        self.show_property(name, cells, display_parameters)

    def show_property(self, name, cells, display_parameters=None):
        '''save a bunch of positions with properties locally and display them'''
        fullpath = joinp(self.output_abs_path, name + '.pts')
        viewer.export_points(fullpath, cells, name)
        self.show(name + '.pts', display_parameters)

    def show_placement(self, name, cells, coloring=None, display_parameters=None):
        '''save a bunch of morphologies placement and display them

        Args:
            coloring: the name of the property to use as color (default: morphology)
        '''
        fullpath = joinp(self.output_abs_path, name + '.placement')
        coloring = coloring if coloring is not None else 'morphology'
        viewer.export_positions_vectors(fullpath, cells, coloring)
        morph_fullpath = joinp(self.output_abs_path, name + '.txt')
        viewer.export_strings(morph_fullpath, cells.properties.morphology)
        self.show(name + '.placement', display_parameters)

    def show_sdist(self, sdist, attribute, value, display_parameters=None):
        '''visualize the 3D probability distribution of the particular value of an attribute
        from a spatial distribution

        Voxels where the probability is missing or zero are not shown.
        '''
        filename = 'sdist_' + value.replace(' ', '_')
        raw = sdist.get_probability_field(attribute, value).astype(np.float32)
        mhd = core.get_mhd_info(raw.shape, np.float32, sdist.voxel_dimensions, filename + '.raw')
        self.show_volume(filename, core.VoxelData(mhd, raw), display_parameters)

    def show_vectors(self, name, field, point_count, voxel_dimensions, display_parameters=None):
        '''visualize a vector field'''
        filename = name + '.vcf'
        viewer.export_vector_field(joinp(self.output_abs_path, filename),
                                   field, point_count, voxel_dimensions)
        self.show(filename, display_parameters)
