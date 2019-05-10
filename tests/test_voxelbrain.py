import os
import json
import nose.tools as nt
import numpy as np
import numpy.testing as npt

from mock import patch

from voxcell import OrientationField, VoxelData, RegionMap
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

    @patch('os.path.exists', return_value=True)
    @patch('voxcell.Hierarchy.load_json', return_value='test')
    def test_load_hierarchy(self, mock, _):
        actual = self.atlas.load_hierarchy()
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/hierarchy.json')

    @patch('os.path.exists', return_value=True)
    @patch('voxcell.RegionMap.load_json', return_value='test')
    def test_load_region_map(self, mock, _):
        actual = self.atlas.load_region_map()
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/hierarchy.json')

    @patch('os.path.exists', return_value=True)
    @patch('voxcell.VoxelData.load_nrrd', return_value='test')
    def test_load_data_1(self, mock, _):
        actual = self.atlas.load_data('bar')
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/bar.nrrd')

    @patch('os.path.exists', return_value=True)
    @patch('voxcell.OrientationField.load_nrrd', return_value='test')
    def test_load_data_2(self, mock, _):
        actual = self.atlas.load_data('bar', cls=OrientationField)
        nt.assert_equal(actual, 'test')
        mock.assert_called_with('/foo/bar.nrrd')

    @patch('os.path.exists', return_value=True)
    @patch('voxcell.VoxelData.load_nrrd')
    @patch('voxcell.RegionMap.load_json')
    def test_get_region_mask(self, mock_rmap, mock_data, _):
        mock_rmap.return_value = RegionMap.from_dict({
            'id': 1,
            'acronym': 'A',
            'name': 'aA',
            'children': [{
                'id': 2,
                'acronym': 'B',
                'name': 'Bb',
            }]
        })
        mock_data.return_value = VoxelData(np.array([1, 0, 2]), [1.], [0.])

        npt.assert_equal(
            self.atlas.get_region_mask('A').raw,
            [True, False, True]
        )
        npt.assert_equal(
            self.atlas.get_region_mask('a', ignore_case=True).raw,
            [True, False, True]
        )
        nt.assert_raises(
            VoxcellError,
            self.atlas.get_region_mask, 'a'
        )
        npt.assert_equal(
            self.atlas.get_region_mask('A', with_descendants=False).raw,
            [True, False, False]
        )
        npt.assert_equal(
            self.atlas.get_region_mask('aa', attr='name',
                                       ignore_case=True).raw,
            [True, False, True]
        )
        nt.assert_raises(
            VoxcellError,
            self.atlas.get_region_mask, 'aa'
        )
        npt.assert_equal(
            self.atlas.get_region_mask('aA', attr='name').raw,
            [True, False, True]
        )
        nt.assert_raises(
            VoxcellError,
            self.atlas.get_region_mask, 'aa'
        )
