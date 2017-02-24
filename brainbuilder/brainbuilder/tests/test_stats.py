import numpy as np
import pandas as pd
import pandas.util.testing as pdt

import neurom as nm

from mock import Mock, call, patch

import brainbuilder.stats as test_module


@patch('neurom.get')
def test_segment_region_histogram(nm_get):
    def lookup_mock(positions, outer_value=None):
        return positions[:, 1].astype(int)

    def nm_get_mock(feature, neurites, **kwargs):
        assert(feature == 'segment_midpoints')
        assert('neurite_type' in kwargs)
        neurite_type = kwargs['neurite_type']
        MOCK_SEGMENT_MIDPOINTS = {
            nm.APICAL_DENDRITE:
                [[0., 0., 0.]],
            nm.AXON:
                [[1., 1., 1.], [3., 3., 3.], [5., 5., 5]],
            nm.BASAL_DENDRITE:
                [],
        }
        return MOCK_SEGMENT_MIDPOINTS[kwargs['neurite_type']]

    cells = pd.DataFrame([
        {
            'x': 100.,
            'y': 100.,
            'z': 100.,
            'orientation': [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
            'morphology': 'morph-X'
        },
    ], index=[42])

    nm_get.configure_mock(**{
        'side_effect': nm_get_mock,
    })

    morphologies = Mock(get=Mock(return_value='nrn'))
    annotation = Mock(lookup=lookup_mock)
    transform = {
        100: 'A',
        101: 'B',
        102: 'B',
        103: 'B',
    }

    result = test_module.segment_region_histogram(cells, morphologies, annotation, transform)
    morphologies.get.assert_called_with('morph-X')
    nm_get.assert_has_calls([
        call('segment_midpoints', 'nrn', neurite_type=nm.APICAL_DENDRITE),
        call('segment_midpoints', 'nrn', neurite_type=nm.AXON),
        call('segment_midpoints', 'nrn', neurite_type=nm.BASAL_DENDRITE),
    ])

    expected = pd.DataFrame(
        [
            [1, 0, 0],
            [0, 2, 1],
            [0, 0, 0],
        ],
        columns=['A', 'B', None],
        index=pd.MultiIndex.from_tuples(
            [(42, 'apical_dendrite'), (42, 'axon'), (42, 'basal_dendrite')], names=['gid', 'branch_type']
        )
    )
    pdt.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1))
