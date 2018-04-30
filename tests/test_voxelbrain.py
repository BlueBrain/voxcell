import os
import nose.tools as nt

from mock import patch

from voxcell import OrientationField
from voxcell.exceptions import VoxcellError

import voxcell.nexus.voxelbrain as test_module


def test_open_atlas():
    nt.assert_is_instance(test_module.Atlas.open('foo'), test_module.LocalAtlas)
    nt.assert_is_instance(test_module.Atlas.open('file://foo'), test_module.LocalAtlas)
    nt.assert_raises(VoxcellError, test_module.Atlas.open, 'foo://bar')
    nt.assert_raises(VoxcellError, test_module.Atlas.open, 'http://foo/bar')


class TestLocalAtlas(object):
    def setUp(self):
        self.atlas = test_module.Atlas.open('/foo')

    def test_class(self):
        nt.assert_is_instance(self.atlas, test_module.LocalAtlas)

    @patch('voxcell.Hierarchy.load_json', return_value='test')
    def test_load_hierarchy(self, mock):
        actual = self.atlas.load_hierarchy()
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/hierarchy.json')

    @patch('voxcell.VoxelData.load_nrrd', return_value='test')
    def test_load_data_1(self, mock):
        actual = self.atlas.load_data('bar')
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/bar.nrrd')

    @patch('voxcell.OrientationField.load_nrrd', return_value='test')
    def test_load_data_2(self, mock):
        actual = self.atlas.load_data('bar', cls=OrientationField)
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/bar.nrrd')
