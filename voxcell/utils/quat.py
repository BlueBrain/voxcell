"""
A wrapper around 'numpy-quaternion' package.

On import, the last one emits a verbose "code may run MUCH more slowly" warning,
which looks pretty scary but is not relevant for our limited usage of this library.

See also:
https://github.com/moble/quaternion#dependencies

We supress this warning to avoid confusing users.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # pylint: disable=unused-import
    from quaternion import (
        as_float_array,
        as_rotation_matrix,
        from_float_array,
        from_rotation_matrix,
    )
