""" Region hierarchy tree. """

import copy
import json
import re
import logging

import six

from voxcell.exceptions import VoxcellError


L = logging.getLogger(__name__)


class Matcher(object):
    """ Helper class for value search. """
    def __init__(self, value, ignore_case=False):
        self.value = value
        if isinstance(value, six.string_types):
            self.ignore_case = ignore_case
            if value.startswith("@"):
                self.value = re.compile(value[1:], re.IGNORECASE if ignore_case else 0)
        else:
            if ignore_case:
                L.warning("Not a string value; ignoring 'ignore_case' flag")

    def __call__(self, value):
        if hasattr(self.value, 'match'):
            return bool(self.value.search(value))
        elif isinstance(value, six.string_types) and self.ignore_case:
            return self.value.upper() == value.upper()
        else:
            return self.value == value


class RegionMap(object):
    """ Region ID <-> attribute mapping. """
    def __init__(self):
        self._data = dict()
        self._children = dict()
        self._parent = dict()

    def get(self, _id, attr, with_ascendants=False):
        """
        Get attribute value associated with region ID.

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
        """
        Find IDs of the regions matching a given attribute.

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

    def _get(self, _id, attr):
        """ Fetch attribute value for a given region ID. """
        if _id not in self._data:
            raise VoxcellError("Region ID not found: %d" % _id)
        node = self._data[_id]
        if attr not in node:
            raise VoxcellError("Attribute not found: '%s' [region ID = %d]" % (attr, _id))
        return node[attr]

    def _ascendants(self, _id):
        """ List of ascendants for a given region ID (itself included; sorted "upwards"). """
        x = _id
        result = []
        while x is not None:
            result.append(x)
            x = self._parent[x]
        return result

    def _descendants(self, _id):
        """ Set of descendants for a given region ID (itself included). """
        result = set([_id])
        for c in self._children[_id]:
            result.update(self._descendants(c))
        return result

    @classmethod
    def from_dict(cls, d):
        """
        Construct RegionMap from a hierarchical dictionary.
        """
        def include(data, parent_id):
            # pylint: disable=protected-access,missing-docstring
            _id = data['id']
            if _id in result._data:
                raise VoxcellError("Duplicate id: %d" % _id)
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
        """
        Construct RegionMap from JSON file.

        Note:
            If top-most object contains 'msg' field, Allen Brain Institute JSON layout is assumed.
        """
        with open(filepath, 'r') as f:
            content = json.load(f)

        if 'msg' in content:
            L.warning("Top-most object contains 'msg'; assuming AIBS JSON layout")
            if len(content['msg']) > 1:
                raise VoxcellError("Unexpected JSON layout (more than one 'msg' child)")
            content = content['msg'][0]

        return cls.from_dict(content)