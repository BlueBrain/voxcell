"""Region hierarchy tree."""

import copy
import json
import logging
import re

from voxcell.exceptions import VoxcellError

L = logging.getLogger(__name__)


class Matcher:
    """Helper class for value search."""
    def __init__(self, value, ignore_case=False):
        """Init Matcher."""
        self.value = value
        if isinstance(value, str):
            self.ignore_case = ignore_case
            if value.startswith("@"):
                self.value = re.compile(value[1:], re.IGNORECASE if ignore_case else 0)
        else:
            if ignore_case:
                L.warning("Not a string value; ignoring 'ignore_case' flag")

    def __call__(self, value):
        """Return True if the given value matches."""
        if hasattr(self.value, 'match'):
            return bool(self.value.search(value))
        elif isinstance(value, str) and self.ignore_case:
            return self.value.upper() == value.upper()
        else:
            return self.value == value


class RegionMap:
    """Region ID <-> attribute mapping."""
    def __init__(self):
        """Init RegionMap."""
        self._data = {}
        self._children = {}
        self._parent = {}

    def get(self, _id, attr, with_ascendants=False):
        """Get attribute value associated with region ID.

        Args:
            _id (int): region ID of interest
            attr (str): attribute of interest
            with_ascendants (bool): collect attribute value upwards the "lineage"

        Returns:
            - if `with_ascendants=False`: attribute value for given region ID
            - otherwise: list of values starting from the "bottom" hierarchy level towards "top"

        Raises:
            - VoxcellError if either region ID or attribute key are can not be found

        Example:
            >>> rmap.get(315, 'name')
            'Isocortex'
        """
        if with_ascendants:
            return [self._get(k, attr) for k in self._ascendants(_id)]
        else:
            return self._get(_id, attr)

    def find(self, value, attr, ignore_case=False, with_descendants=False):
        """Find IDs of the regions matching a given attribute.

        Args:
            value: attribute value to match
            attr (str): attribute of interest
            ignore_case (bool): ignore case (when comparing strings)
            with_descendants (bool): collect region IDs downwards the "lineage"

        If `value` starts with '@' symbol, `value[1:]` is used a regular expression.
        Any substring matching the regular expression would be matched;
        please used '^' and '$' for "starts with" or "ends with" restrictions.

        Regular expressions can be used together with `ignore_case`.

        Returns:
            - if `with_descendants=False`: set of IDs of the regions matching the attribute
            - otherwise: set of region IDs matching the attribute + all their children recursively

        Example:
            >>> rmap.find("@layer 1", attr='name', ignore_case=True, with_descendants=True)
            set([1, 2, 4, 5])
        """
        matcher = Matcher(value, ignore_case=ignore_case)

        result = set()
        for _id in self._data:
            if matcher(self._get(_id, attr)):
                if with_descendants:
                    result.update(self._descendants(_id))
                else:
                    result.add(_id)

        return result

    def is_leaf_id(self, _id):
        """Indicate whether or not the input identifier is a leaf of the hierarchy tree.

        A leaf identifier is the identifier of a region with no children.

        Args:
            _id(int): region identifier, i.e., an 'id' value in hierarchy.json.

        Returns:
            True, if is a leaf, False otherwise.

        Raises:
            VoxcellError if the identifier cannot be found.

        Example:
            >>> rmap.is_leaf_id(399)
            True
            >>> rmap.is_leaf_id(-10)
            VoxcellError: Region ID not found: -10
        """
        if _id not in self._data:
            raise VoxcellError(f"Region ID not found: {_id}")
        return not self._children[_id]

    def _get(self, _id, attr):
        """Fetch attribute value for a given region ID."""
        if _id not in self._data:
            raise VoxcellError(f"Region ID not found: {_id}")
        node = self._data[_id]
        if attr not in node:
            raise VoxcellError(f"Attribute not found: '{attr}' [region ID = {_id}]")
        return node[attr]

    def _ascendants(self, _id):
        """List of ascendants for a given region ID (itself included; sorted "upwards")."""
        x = _id
        result = []
        while x is not None:
            result.append(x)
            x = self._parent[x]
        return result

    def _descendants(self, _id):
        """Set of descendants for a given region ID (itself included)."""
        result = set([_id])
        for c in self._children[_id]:
            result.update(self._descendants(c))
        return result

    @classmethod
    def from_dict(cls, d):
        """Construct RegionMap from a hierarchical dictionary."""
        def include(data, parent_id):
            # pylint: disable=protected-access,missing-docstring
            _id = data['id']
            if _id in result._data:
                raise VoxcellError(f"Duplicate id: {_id}")
            children = data.pop('children', [])
            result._data[_id] = data
            result._parent[_id] = parent_id
            result._children[_id] = [c['id'] for c in children]
            for c in children:
                include(c, _id)
        result = cls()
        include(copy.deepcopy(d), None)
        return result

    @classmethod
    def load_json(cls, filepath):
        """Construct RegionMap from JSON file.

        Note:
            If top-most object contains 'msg' field, Allen Brain Institute JSON layout is assumed.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)

        if 'msg' in content:
            if len(content['msg']) > 1:
                raise VoxcellError("Unexpected JSON layout (more than one 'msg' child)")
            content = content['msg'][0]

        return cls.from_dict(content)
