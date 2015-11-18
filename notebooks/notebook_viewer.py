'''simple integration of the brain builder viewer with ipython notebooks'''

import os
import numpy as np
from os.path import join as joinp
from brainbuilder.utils import viewer
from brainbuilder.utils import genbrain as gb
from IPython.display import HTML, display


class NotebookViewer(object):
    '''encapsulates the integration of the webgl in an ipython notebook'''

    def __init__(self):
        ''' NotebookViewer '''
        # The location where the output will be stored
        # under the viewer directory
        self.directory = 'out'
        self.output_directory = joinp('..', 'viewer', self.directory)

        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

    def show(self, filename):
        '''display the given local file'''
        url = '../../static/index.html#' + joinp(self.directory, filename)
        html = (
            '<iframe src="{url}"'
            ' scrolling="no" width="700" height="350" allowfullscreen></iframe>\n'
        ).format(url=url)

        display(HTML(html))

    def show_volume(self, name, voxel):
        '''save a VoxelData object locally and display it'''
        filename_mhd = name + '.mhd'
        filename_raw = name + '.raw'
        voxel.save_metaio(joinp(self.output_directory, filename_mhd), filename_raw)
        self.show(filename_mhd)

    def show_points(self, name, cells):
        '''save a bunch of positions locally and display them'''
        self.show_property(name, cells)

    def show_property(self, name, cells):
        '''save a bunch of positions with properties locally and display them'''
        fullpath = joinp(self.output_directory, name + '.pts')
        viewer.export_points(fullpath, cells, name)
        self.show(name + '.pts')

    def show_placement(self, name, cells, coloring=None):
        '''save a bunch of morphologies placement and display them

        Args:
            coloring: the name of the property to use as color (default: morphology)
        '''
        fullpath = joinp(self.output_directory, name + '.placement')
        coloring = coloring if coloring is not None else 'morphology'
        viewer.export_positions_vectors(fullpath, cells, coloring)
        morph_fullpath = joinp(self.output_directory, name + '.txt')
        viewer.export_strings(morph_fullpath, cells.properties.morphology)
        self.show(name + '.placement')

    def show_sdist(self, sdist, attribute, value):
        '''visualize the 3D probability distribution of the particular value of an attribute
        from a spatial distribution

        Voxels where the probability is missing or zero are not shown.
        '''
        filename = 'sdist_' + value.replace(' ', '_')
        raw = sdist.get_probability_field(attribute, value).astype(np.float32)
        mhd = gb.get_mhd_info(raw.shape, np.float32, sdist.voxel_dimensions, filename + '.raw')
        self.show_volume(filename, gb.VoxelData(mhd, raw))

    def show_vectors(self, name, field, point_count, voxel_dimensions):
        '''visualize a vector field'''
        filename = name + '.vcf'
        viewer.export_vector_field(joinp(self.output_directory, filename),
                                   field, point_count, voxel_dimensions)
        self.show(filename)
