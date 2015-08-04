from nose.tools import eq_
import numpy as np
from numpy.testing import assert_equal

from brainbuilder.assignment_synapse_class import assign_synapse_class_randomly


def test_assign_sypnapse_class_empty():
    sclass = assign_synapse_class_randomly(np.zeros(shape=(0, 3)), inhibitory_fraction=0)
    eq_(sclass.shape, (0,))


def test_assign_sypnapse_class_all_excitatory():
    result = assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=0)
    eq_(result.shape, (10,))
    assert_equal(result, np.array(['excitatory'] * 10))


def test_assign_sypnapse_class_all_inhibitory():
    result = assign_synapse_class_randomly(np.zeros(shape=(10, 3)), inhibitory_fraction=1)
    eq_(result.shape, (10,))
    assert_equal(result, np.array(['inhibitory'] * 10))
