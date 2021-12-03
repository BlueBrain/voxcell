from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest

import voxcell.nexus.voxelbrain as test_module
from voxcell import OrientationField, RegionMap, VoxelData
from voxcell.exceptions import VoxcellError

DATA_PATH = Path(__file__).parent / 'data'
DUMMY_ATLAS = test_module.Atlas.open('/foo')


def test_open_atlas():
    assert isinstance(test_module.Atlas.open('foo'), test_module.LocalAtlas)
    assert isinstance(test_module.Atlas.open('file://foo'), test_module.LocalAtlas)
    with pytest.raises(VoxcellError):
        test_module.Atlas.open('foo://bar')
    with pytest.raises(VoxcellError):
        test_module.Atlas.open('http://foo/bar')


@patch('os.path.exists', return_value=True)
@patch('voxcell.RegionMap.load_json', return_value='test')
def test_load_region_map(mock, _):
    actual = DUMMY_ATLAS.load_region_map()
    assert actual == 'test'
    mock.assert_called_with('/foo/hierarchy.json')


@patch('os.path.exists', return_value=True)
@patch('voxcell.nexus.voxelbrain.open')
def test_load_metadata(open_mock, _):
    with open(str(Path(DATA_PATH, 'metadata.json')), 'r') as metadata_file:
        open_mock.return_value = metadata_file
        metadata = DUMMY_ATLAS.load_metadata()
        if 'layers' in metadata:
            assert all(key in metadata['layers'] for key in ['names', 'queries', 'attribute'])


@patch('os.path.exists', return_value=True)
@patch('voxcell.VoxelData.load_nrrd', return_value='test')
def test_load_data_1(mock, _):
    actual = DUMMY_ATLAS.load_data('bar')
    assert actual == 'test'
    mock.assert_called_with('/foo/bar.nrrd')


@patch('os.path.exists', return_value=True)
@patch('voxcell.OrientationField.load_nrrd', return_value='test')
def test_load_data_2(mock, _):
    actual = DUMMY_ATLAS.load_data('bar', cls=OrientationField)
    assert actual == 'test'
    mock.assert_called_with('/foo/bar.nrrd')


@patch('os.path.exists', return_value=True)
@patch('voxcell.VoxelData.load_nrrd')
@patch('voxcell.RegionMap.load_json')
def test_get_region_mask(mock_rmap, mock_data, _):
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
        DUMMY_ATLAS.get_region_mask('A').raw,
        [True, False, True]
    )
    npt.assert_equal(
        DUMMY_ATLAS.get_region_mask('a', ignore_case=True).raw,
        [True, False, True]
    )
    with pytest.raises(VoxcellError):
        DUMMY_ATLAS.get_region_mask('a')
    npt.assert_equal(
        DUMMY_ATLAS.get_region_mask('A', with_descendants=False).raw,
        [True, False, False]
    )
    npt.assert_equal(
        DUMMY_ATLAS.get_region_mask('aa', attr='name',
                                   ignore_case=True).raw,
        [True, False, True]
    )
    with pytest.raises(VoxcellError):
        DUMMY_ATLAS.get_region_mask('aa')
    npt.assert_equal(
        DUMMY_ATLAS.get_region_mask('aA', attr='name').raw,
        [True, False, True]
    )
    with pytest.raises(VoxcellError):
        DUMMY_ATLAS.get_region_mask('aa')


expected_data = (('layer 1', {1140, 1125}),
                 ('layer 2', {1141, 1126}),
                 ('layer 3', {517, 1142, 1127}),
                 ('layer 4', {1128}),
                 ('layer 5', {1129}),
                 ('layer 6', {1130}),
                 ('L1', {1125}),)
expected_names = [v[0] for v in expected_data]
expected_ids = [v[1] for v in expected_data]


def test_get_layer():
    atlas = test_module.Atlas.open(str(DATA_PATH))
    for i in range(7):
        layer, ids = atlas.get_layer(i)
        assert expected_names[i] == layer
        assert expected_ids[i] == ids


@patch('voxcell.nexus.voxelbrain.LocalAtlas.load_region_map', return_value={})
@patch('voxcell.nexus.voxelbrain.LocalAtlas.load_metadata')
def test_get_layer_ids_raise(load_metadata_mock, _):
    atlas = test_module.Atlas.open(str(DATA_PATH))
    load_metadata_mock.return_value = {'layers': {'name': ['layer1', 'layer4']}}
    with pytest.raises(VoxcellError):
        atlas.get_layer(0)

    load_metadata_mock.return_value = \
        {'layers': {'names': ['layer1', 'layer4'], 'queries': ['@.*1$', '@.*4$']}}
    with pytest.raises(VoxcellError):
        atlas.get_layer(1)

    load_metadata_mock.return_value = \
        {'layers': {'names': ['layer1', 'layer4'], 'queries': ['@.*4$'], 'attribute': 'name'}}
    with pytest.raises(VoxcellError):
        atlas.get_layer(2)


def test_get_layer_ids():
    atlas = test_module.Atlas.open(str(DATA_PATH))
    names, ids = atlas.get_layers()
    npt.assert_equal(names, expected_names)
    for id_, expected in zip(ids, expected_ids):
        assert id_ == expected


@patch('voxcell.nexus.voxelbrain.LocalAtlas.load_data')
def test_get_layer_volume(load_data_mock):
    atlas = test_module.Atlas.open(str(DATA_PATH))
    load_data_mock.return_value = VoxelData(
        np.array([[[1125], [1126], [1130]]]), voxel_dimensions=(1.0, 1.0, 1.0))

    layer_volume = atlas.get_layer_volume()
    # In metadata.json, 'L1' and '@.*1' defines the same set of ids.
    # L1 wins because it comes at the end.
    npt.assert_array_equal(layer_volume.raw, np.array([[[7], [2], [6]]]))
