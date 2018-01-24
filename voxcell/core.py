'''core container classes'''

from voxcell.utils import deprecate

deprecate.fail("""
    voxcell.core is deprecated.
    Please change your imports as following:
        from voxcell.core import X -> from voxcell import X
""")
