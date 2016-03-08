from nose.tools import eq_
import numpy as np
from numpy.testing import assert_equal

import brainbuilder.assignment_synapse_class as ASC


def test_assign_synapse_class_empty():
    chosen = ASC.assign_synapse_class_randomly(np.zeros(shape=(0, 3)), inhibitory_fraction=0)
    eq_(chosen.shape, (0, 1))


def test_assign_synapse_class_all_excitatory():
    result = ASC.assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=0)
    eq_(result.shape, (10, 1))
    assert_equal(result.synapse_class, np.array(['EXC'] * 10))


def test_assign_synapse_class_all_inhibitory():
    result = ASC.assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=1)
    eq_(result.shape, (10, 1))
    assert_equal(result.synapse_class, np.array(['INH'] * 10))
