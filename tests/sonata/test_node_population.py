import nose.tools as nt

from voxcell.utils import deprecate
from voxcell.sonata import NodePopulation


def test_init():
    with nt.assert_raises(deprecate.VoxcellDeprecationError):
        NodePopulation(1, 1)
