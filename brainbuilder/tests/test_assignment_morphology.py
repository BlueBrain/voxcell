import tempfile
import numpy as np

from numpy.testing import assert_equal

import brainbuilder.assignment_morphology as AM


def test_serialize():
    morphologies = np.array(['morph0', 'morph1', np.nan, 'morph3'])
    with tempfile.NamedTemporaryFile() as f:
        AM.serialize_assign_morphology(f.name, morphologies)
        new_morphologies = AM.deserialize_assign_morphology(f.name)
        assert_equal(morphologies, new_morphologies)
