import tempfile

import numpy as np
from numpy.testing import assert_equal

import brainbuilder.assignment_orientation as AO


def test_serialization():
    orientations = []
    orientations.append(np.array([[-0.05623054,  0.19198307, -0.50127894],
                                  [ 0.00479315,  0.16031244, -0.39549953],
                                  [-0.04257896,  0.23532444, -0.74958652],
                                  ], dtype=np.float32))
    orientations.append(np.array([[ 0.1075616 ,  0.34998584, -0.57636482],
                                  [ 0.08289455,  0.52759528, -0.82902694],
                                  [ 0.06194865,  0.31040394, -0.68430364],
                                  ], dtype=np.float32))
    orientations.append(np.array([[ 0.0185358 ,  0.30430049, -0.7591911 ],
                                  [ 0.0185358 ,  0.30430049, -0.7591911 ],
                                  [ 0.04053133,  0.3425234 ,  0.81993151],
                                  ], dtype=np.float32))

    with tempfile.NamedTemporaryFile() as f:
        AO.serialize_assigned_orientations(f.name, orientations)
        new_orientations = AO.deserialize_assigned_orientations(f.name)
        for o, new_o in zip(orientations, new_orientations):
            assert_equal(o, new_o)
