import os

import tempfile
import numpy as np

from numpy.testing import assert_equal
from nose.tools import eq_, ok_

import brainbuilder.assignment_metype as AM


positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], ])
mtypes = ['mtype0', 'mtype1', 'mtype2', ]
etypes = ['etype0', 'etype1', 'etype2', ]


def test_assign_metype_random():
    np.random.seed(0)
    metypes = AM.assign_metype_random(positions, mtypes, etypes)

    eq_(len(metypes), 5)
    ok_(all(mtype in mtypes for mtype in metypes[:, 0]))
    ok_(all(etype in etypes for etype in metypes[:, 1]))
    res = np.array([['mtype1', 'etype2'],
                    ['mtype0', 'etype0'],
                    ['mtype1', 'etype0'],
                    ['mtype1', 'etype0'],
                    ['mtype2', 'etype1']], dtype=object)
    assert_equal(metypes, res)


def test_serialize():
    metypes = AM.assign_metype_random(positions, mtypes, etypes)
    with tempfile.NamedTemporaryFile() as f:
        AM.serialize_assign_metype(f.name, metypes)
        new_metypes = AM.deserialze_assign_metype(f.name)
        assert_equal(metypes, new_metypes)
