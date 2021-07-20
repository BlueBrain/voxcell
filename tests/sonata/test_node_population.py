import pytest

from voxcell.utils import deprecate
from voxcell.sonata import NodePopulation


def test_init():
    with pytest.raises(deprecate.VoxcellDeprecationError):
        NodePopulation(1, 1)
