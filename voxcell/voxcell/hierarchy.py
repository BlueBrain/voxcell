""" Region hierarchy tree. """

import copy
import json
import itertools

from six import iteritems

from voxcell import deprecate
from voxcell.exceptions import VoxcellError


def _sub_dict(dict_, keys):
    """ Get a dictionary for subset of keys. """
    return {k: dict_[k] for k in keys}


class Hierarchy(object):
    '''encapsulates data about brain structures organized in a
    hierarchical part-of relationship.'''

    def __init__(self, data):
        self.children = [Hierarchy(c) for c in data.get('children', [])]
        self.data = dict((k, data[k]) for k in set(data.keys()) - set(['children']))

    @classmethod
    def load(cls, filename):
        '''load a hierarchy of annotations in json from the Allen Brain Institute'''
        with open(filename, 'r') as f:
            return Hierarchy(json.load(f)['msg'][0])

    def find(self, attribute, value):
        '''get a list with all the subsections of a hierarchy that exactly match
        the given value on the given attribute'''
        me = [self] if self.data[attribute] == value else []
        children = (c.find(attribute, value) for c in self.children)
        return list(itertools.chain(me, *children))

    def get(self, attribute):
        '''get the set of all values of the given attribute for every subsection of a hierarchy'''
        me = [self.data[attribute]]
        children = (c.get(attribute) for c in self.children)
        return set(itertools.chain(me, *children))

    def collect(self, find_attribute, value, get_attribute):
        '''get the set of all values for the get_attribute for all sections in a hierarchy
        matching the find_attribute-value'''
        collected = (r.get(get_attribute) for r in self.find(find_attribute, value))
        return set(itertools.chain.from_iterable(collected))

    def __str__(self):
        txt = self.data.get('name', '<unnamed section>')

        if self.data:
            txt += '\n    ' + '\n    '.join('%s: %s' % (k, v)
                                            for k, v in iteritems(self.data) if k != 'name')

        if self.children:
            children = '\n'.join(str(c) for c in self.children)
            txt += '\n    children: [\n    %s\n    ]' % children.replace('\n', '\n    ')
        else:
            txt += '\n    children: []'

        return txt


class RegionMap(object):
    """ Encapsulates brain region hierarchy retrieved from NIP. """
    def __init__(self):
        self._data = dict()
        self._children = dict()
        self._parent = dict()

    def get(self, _id):
        """ Get the payload data associated with region ID. """
        return self._data[_id]

    @classmethod
    def from_dict(cls, d):
        """ Construct RegionMap from a hierarchical dictionary. """
        def include(data, parent_id):
            # pylint: disable=protected-access,missing-docstring
            _id = data.pop('id')
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
        """ Construct RegionMap from JSON file. """
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_json(cls, filepath):
        """ Deprecated. """
        deprecate.warn("Method has been renamed. Please use load_json() instead.")
        return cls.load_json(filepath)

    def ids(self, value, attr='name', with_descendants=True):
        """ Get set of regions matching the given attribute. """
        result = set()
        for _id, data in iteritems(self._data):
            if data.get(attr) == value:
                if with_descendants:
                    result.update(self.descendants(_id))
                else:
                    result.add(_id)
        return result

    def ascendants(self, _id):
        """ Get a list of ascendants for the given id (itself included).

            The list is sorted in level descending order.
        """
        x = _id
        result = []
        while x is not None:
            result.append(x)
            x = self._parent[x]
        return result

    def descendants(self, _id):
        """ Get a set of descendants for the given id (itself included). """
        result = set([_id])
        for c in self._children[_id]:
            result.update(self.descendants(c))
        return result

    def sub(self, _id):
        """ Return a RegionMap of region for the given id. """
        ids = self.descendants(_id)
        result = self.__class__()
        # pylint: disable=protected-access
        result._data = _sub_dict(self._data, ids)
        result._children = _sub_dict(self._children, ids)
        result._parent = _sub_dict(self._parent, ids)
        return result
