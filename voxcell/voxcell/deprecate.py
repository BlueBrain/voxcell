""" Deprecation utilities. """

import warnings

from voxcell import VoxcellError


class VoxcellDeprecationWarning(UserWarning):
    """ voxcell deprecation warning. """
    pass


class VoxcellDeprecationError(VoxcellError):
    """ voxcell deprecation error. """
    pass


def fail(msg=None):
    """ Raise a deprecation exception. """
    raise VoxcellDeprecationError(msg)


def warn(msg=None):
    """ Issue a deprecation warning. """
    warnings.warn(msg, VoxcellDeprecationWarning)
