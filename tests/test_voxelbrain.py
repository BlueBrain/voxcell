from pathlib import Path
import json
import nose.tools as nt
import numpy as np
import numpy.testing as npt

DATA_PATH = Path(Path(__file__).parent, 'data')

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
        with open(str(Path(DATA_PATH, 'metadata.json')), 'r') as metadata_file:
            self.metadata = json.load(metadata_file)
        self.hierarchy = RegionMap.load_json(str(Path(DATA_PATH, 'hierarchy.json')))

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
    @patch('voxcell.nexus.voxelbrain.open')
    def test_load_metadata(self, open_mock, _):
        with open(str(Path(DATA_PATH, 'metadata.json')), 'r') as metadata_file:
            open_mock.return_value = metadata_file
            metadata = self.atlas.load_metadata()
            if 'layers' in metadata:
                assert all(key in metadata['layers'] for key in ['names', 'queries', 'attribute'])

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


class TestLocalAtlas(object):
    def setUp(self):
        self.atlas = test_module.Atlas.open(str(DATA_PATH))
        expected_data = (('layer 1', set([1140, 1125])),
                         ('layer 2', set([1141, 1126])),
                         ('layer 3', set([517, 1142, 1127])),
                         ('layer 4', set([1128])),
                         ('layer 5', set([1129])),
                         ('layer 6', set([1130])),
                         ('L1', set([1125])),
                         )
        self.expected_names = [v[0] for v in expected_data]
        self.expected_ids = [v[1] for v in expected_data]

    def test_get_layer(self):
        for i in range(7):
            layer, ids = self.atlas.get_layer(i)
            assert self.expected_names[i]== layer
            assert self.expected_ids[i] == ids

    @patch('voxcell.nexus.voxelbrain.LocalAtlas.load_region_map', return_value={})
    @patch('voxcell.nexus.voxelbrain.LocalAtlas.load_metadata')
    def test_get_layer_ids_raise(self, load_metadata_mock, _):
        load_metadata_mock.return_value = {'layers': {'name': ['layer1', 'layer4']}}
        with nt.assert_raises(VoxcellError):
            self.atlas.get_layer(0)

        load_metadata_mock.return_value = \
            {'layers': {'names': ['layer1', 'layer4'], 'queries': ['@.*1$', '@.*4$']}}
        with nt.assert_raises(VoxcellError):
            self.atlas.get_layer(1)

        load_metadata_mock.return_value = \
            {'layers': {'names': ['layer1', 'layer4'], 'queries': ['@.*4$'], 'attribute': 'name'}}
        with nt.assert_raises(VoxcellError):
            self.atlas.get_layer(2)


    def test_get_layer_ids(self):
        names, ids = self.atlas.get_layers()
        npt.assert_equal(names, self.expected_names)
        for id_, expected in zip(ids,  self.expected_ids):
            assert id_ == expected

    @patch('voxcell.nexus.voxelbrain.LocalAtlas.load_data')
    def test_get_layer_volume(self, load_data_mock):
        load_data_mock.return_value = VoxelData(
            np.array([[[1125], [1126], [1130]]]), voxel_dimensions=(1.0, 1.0, 1.0))

        layer_volume = self.atlas.get_layer_volume()
        # In metadata.json, 'L1' and '@.*1' defines the same set of ids.
        # L1 wins because it comes at the end.
        npt.assert_array_equal(layer_volume.raw, np.array([[[7], [2], [6]]]))
