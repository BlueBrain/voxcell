""" Region hierarchy tree. """

import json
import itertools

from six import iteritems

from voxcell.utils import deprecate


class Hierarchy(object):
    '''encapsulates data about brain structures organized in a
    hierarchical part-of relationship.'''

    def __init__(self, data):
        deprecate.warn("Deprecated. Please use RegionMap instead. Will be removed in v2.8.0.")
        self.children = [Hierarchy(c) for c in data.get('children', [])]
        self.data = dict((k, data[k]) for k in set(data.keys()) - set(['children']))

    @classmethod
    def load(cls, filename):
        '''load a hierarchy of annotations in json from the Allen Brain Institute'''
        deprecate.warn("Deprecated. Please use load_json() instead.")
        with open(filename, 'r') as f:
            return Hierarchy(json.load(f)['msg'][0])

    @classmethod
    def load_json(cls, filepath):
        """ Construct Hierarchy from VoxelBrain region hierarchy JSON. """
        with open(filepath, 'r') as f:
            return cls(json.load(f))

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
