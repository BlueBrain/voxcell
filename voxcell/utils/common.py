"""Common utils."""
from voxcell.exceptions import VoxcellError


def all_equal(collection):
    """Return True if all the items are equal, False otherwise."""
    prev = None
    for n, item in enumerate(collection):
        if n > 0 and item != prev:
            return False
        prev = item
    return True


def safe_update(d, keys, value):
    """Update the dict and its nested dicts with the given value.

    Raises:
        VoxcelError if the key already exists and the value is different.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    key = keys[-1]
    if key not in d:
        d[key] = value
    elif d[key] != value:
        raise VoxcellError(
            f"Cannot overwrite existing key {key!r} "
            f"having value {d[key]!r} with the new value {value!r}"
        )
