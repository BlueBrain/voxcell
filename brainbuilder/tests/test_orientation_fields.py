import itertools
import tempfile

import numpy as np
from numpy.testing import assert_equal

import brainbuilder.orientation_fields as OF


def test_serialization():
    dim = (10, 10, 10) # more realistic is (528, 320, 456), but that's slow
    np.random.seed(0)
    right = [np.random.rand(*dim), np.random.rand(*dim), np.random.rand(*dim), ]
    fwd = [np.random.rand(*dim), np.random.rand(*dim), np.random.rand(*dim), ]
    up = [np.random.rand(*dim), np.random.rand(*dim), np.random.rand(*dim), ]

    orientation_field = {'right': right, 'up': up, 'fwd': fwd}
    with tempfile.NamedTemporaryFile() as f:
        OF.serialize_orientation_fields(f.name, orientation_field)
        new_orientation_field = OF.deserialize_orientation_fields(f.name)
        for name, i in itertools.product(OF.VECTOR_NAMES, range(3)):
            assert_equal(orientation_field[name][i], new_orientation_field[name][i])
