"""Deprecation utilities."""

import warnings

from voxcell.exceptions import VoxcellError


class VoxcellDeprecationWarning(UserWarning):
    """Voxcell deprecation warning."""


class VoxcellDeprecationError(VoxcellError):
    """Voxcell deprecation error."""


def fail(msg=None):
    """Raise a deprecation exception."""
    raise VoxcellDeprecationError(msg)


def warn(msg=None):
    """Issue a deprecation warning."""
    warnings.warn(msg, VoxcellDeprecationWarning)
