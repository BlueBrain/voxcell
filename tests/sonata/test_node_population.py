import pytest

from voxcell.sonata import NodePopulation
from voxcell.utils import deprecate


def test_init():
    with pytest.raises(deprecate.VoxcellDeprecationError):
        NodePopulation(1, 1)
