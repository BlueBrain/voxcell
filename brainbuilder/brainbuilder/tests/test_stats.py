import numpy as np
import pandas as pd
import pandas.util.testing as pdt

import neurom as nm

from mock import Mock, call, patch

import brainbuilder.stats as test_module


def _mock_lookup(positions, outer_value=None):
    return positions[:, 1].astype(int)


def _mock_nm_get(feature, neurites, **kwargs):
    MOCK_SEGMENT_MIDPOINTS = {
        nm.APICAL_DENDRITE:
            [[0., 0., 0.]],
        nm.AXON:
            [[1., 1., 1.], [1., 1., 1.], [3., 3., 3.], [5., 5., 5]],
        nm.BASAL_DENDRITE:
            [],
    }
    MOCK_SEGMENT_VOLUMES = {
        nm.APICAL_DENDRITE:
            [1.],
        nm.AXON:
            [2., 3., 4., 5.],
        nm.BASAL_DENDRITE:
            []
    }
    if feature == 'segment_midpoints':
        return MOCK_SEGMENT_MIDPOINTS[kwargs['neurite_type']]
    elif feature == 'segment_volumes' or feature == 'segment_lengths':
        return MOCK_SEGMENT_VOLUMES[kwargs['neurite_type']]



class Test_segment_region_histogram(object):
    def setUp(self):
        self.patcher = patch('neurom.get')
        self.nm_get = self.patcher.start()
        self.nm_get.configure_mock(**{
            'side_effect': _mock_nm_get,
        })

        self.morphologies = Mock(get=Mock(return_value='nrn'))
        self.annotation = Mock(lookup=_mock_lookup)

        self.cells = pd.DataFrame([
            {
                'x': 100.,
                'y': 100.,
                'z': 100.,
                'orientation': [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                'morphology': 'morph-X'
            },
        ], index=[42])


    def tearDown(self):
        self.patcher.stop()


    def test_basic(self):
        result = test_module.segment_region_histogram(
            self.cells, self.morphologies, self.annotation
        )

        self.morphologies.get.assert_called_with('morph-X')
        self.nm_get.assert_has_calls([
            call('segment_midpoints', 'nrn', neurite_type=nm.APICAL_DENDRITE),
            call('segment_midpoints', 'nrn', neurite_type=nm.AXON),
            call('segment_midpoints', 'nrn', neurite_type=nm.BASAL_DENDRITE),
        ])

        expected = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [0, 2, 1, 1],
                [0, 0, 0, 0],
            ],
            columns=[100, 101, 103, 105],
            index=pd.MultiIndex.from_tuples(
                [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


    def test_annotation_transform(self):
        transform = {
            100: 'A',
            101: 'B',
            102: 'B',
            103: 'B',
            300: 'C',
        }

        result = test_module.segment_region_histogram(
            self.cells, self.morphologies, self.annotation, transform
        )

        expected = pd.DataFrame(
            [
                [1, 0, 0, 0],
                [0, 3, 0, 1],
                [0, 0, 0, 0],
            ],
            columns=['A', 'B', 'C', None],
            index=pd.MultiIndex.from_tuples(
                [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
            )
        )

        # pandas does not sort reliably index containing None
        result.rename(columns={None: "None"}, inplace=True)
        expected.rename(columns={None: "None"}, inplace=True)

        pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


    def test_normalize(self):
        result = test_module.segment_region_histogram(
            self.cells, self.morphologies, self.annotation, normalize=True
        )

        expected = pd.DataFrame(
            [
                [1., 0, 0, 0],
                [0, 0.5, 0.25, 0.25],
                [None, None, None, None],
            ],
            columns=[100, 101, 103, 105],
            index=pd.MultiIndex.from_tuples(
                [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


    def test_by_length(self):
        result = test_module.segment_region_histogram(
            self.cells, self.morphologies, self.annotation, by='length'
        )

        expected = pd.DataFrame(
            [
                [1., 0., 0., 0.],
                [0., 5., 4., 5.],
                [0., 0., 0., 0.],
            ],
            columns=[100, 101, 103, 105],
            index=pd.MultiIndex.from_tuples(
                [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))


    def test_by_volume(self):
        result = test_module.segment_region_histogram(
            self.cells, self.morphologies, self.annotation, by='volume'
        )

        expected = pd.DataFrame(
            [
                [1., 0., 0., 0.],
                [0., 5., 4., 5.],
                [0., 0., 0., 0.],
            ],
            columns=[100, 101, 103, 105],
            index=pd.MultiIndex.from_tuples(
                [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
            )
        )
        pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))
