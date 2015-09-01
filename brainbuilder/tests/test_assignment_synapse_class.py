import tempfile

from nose.tools import eq_
import numpy as np
from numpy.testing import assert_equal

import brainbuilder.assignment_synapse_class as ASC


def test_assign_synapse_class_empty():
    sclass = ASC.assign_synapse_class_randomly(np.zeros(shape=(0, 3)), inhibitory_fraction=0)
    eq_(sclass.shape, (0,))


def test_assign_synapse_class_all_excitatory():
    result = ASC.assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=0)
    eq_(result.shape, (10,))
    assert_equal(result, np.array(['excitatory'] * 10))


def test_assign_synapse_class_all_inhibitory():
    result = ASC.assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=1)
    eq_(result.shape, (10,))
    assert_equal(result, np.array(['inhibitory'] * 10))


def test_serialization():
    sclasses = ASC.assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=1)
    sclasses[0] = np.nan

    with tempfile.NamedTemporaryFile() as f:
        ASC.serialize_assign_synapse_class(f.name, sclasses)
        new_sclasses = ASC.deserialize_assign_synapse_class(f.name)
        assert_equal(sclasses, new_sclasses)