"""
SONATA Nodes access / writer.

A thin wrapper around `libsonata` with some syntactic sugar and Pandas view.

See also:
        https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#neuron_networks_nodes

TODO:
    Use `libsonata` instead, once it provides *write* functionality (?)
"""

import h5py
import numpy as np
import pandas as pd
import six

from libsonata import NodeStorage, Selection

from voxcell.exceptions import VoxcellError
from voxcell.math_utils import euler2mat, mat2euler


def _open_population(h5_filepath):
    storage = NodeStorage(h5_filepath)
    populations = storage.population_names
    if len(populations) != 1:
        raise VoxcellError(
            "Only single-population node collections are supported (found: %d)" % len(populations)
        )
    return storage.open_population(list(populations)[0])


class DataFrameProxy(pd.DataFrame):
    """ Pandas DataFrame with some restrictions on adding columns. """
    def __init__(self, size):
        super(DataFrameProxy, self).__init__(index=np.arange(size))

    def __setitem__(self, key, value):
        if key == 'dynamics_params':
            raise VoxcellError("'%s' attribute name is reserved" % key)
        if key.startswith('@'):
            raise VoxcellError("'@' attribute names are reserved")
        if key in self:
            raise VoxcellError("Can not overwrite attribute: '%s'" % key)
        super(DataFrameProxy, self).__setitem__(key, value)


class NodePopulation(object):
    """ Read / write access to one-population on-group SONATA node collection. """
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self._attributes = DataFrameProxy(size)
        self._dynamics_attributes = DataFrameProxy(size)

    def to_dataframe(self):
        """
        Convert into pandas DataFrame.

        Attributes and dynamics attributes are merged into single DataFrame.
        Attribute names for the latter ones are prefixed with '@dynamics'.
        """
        result = pd.DataFrame()
        for prop, column in self._attributes.iteritems():
            result[prop] = column.copy()
        for prop, column in self._dynamics_attributes.iteritems():
            result['@dynamics:%s' % prop] = column.copy()
        return result

    @property
    def positions(self):
        """ N x 3 NumPy array with cell positions. """
        return self._attributes[['x', 'y', 'z']].values

    @positions.setter
    def positions(self, values):
        """ Set position attributes from single N x 3 NumPy array. """
        self._attributes['x'] = values[:, 0]
        self._attributes['y'] = values[:, 1]
        self._attributes['z'] = values[:, 2]

    @property
    def orientations(self):
        """ N x 3 x 3 NumPy array with cell orientations. """
        def _get_angles(ax):
            prop = 'rotation_angle_{ax}axis'.format(ax=ax)
            if prop in self._attributes:
                return self._attributes[prop].values
            else:
                return np.zeros(self.size)
        return euler2mat(
            _get_angles('z'),
            _get_angles('y'),
            _get_angles('x')
        )

    @orientations.setter
    def orientations(self, values):
        """ Set rotation attributes from single N x 3 x 3 NumPy array. """
        az, ay, ax = mat2euler(values)
        self._attributes['rotation_angle_xaxis'] = ax
        self._attributes['rotation_angle_yaxis'] = ay
        self._attributes['rotation_angle_zaxis'] = az

    @property
    def attributes(self):
        """ Attributes as GID-based Pandas DataFrame. """
        return self._attributes

    @property
    def dynamics_attributes(self):
        """ Dynamics attributes as GID-based Pandas DataFrame. """
        return self._dynamics_attributes

    @classmethod
    def load(cls, filepath):
        """
        Load NodePopulation from SONATA Nodes HDF5.

        Limitations:
          - no CSV support
          - one population per file
          - one group per population
        """
        nodes = _open_population(filepath)

        result = cls(nodes.name, nodes.size)

        _all = Selection([(0, nodes.size)])

        for prop in sorted(nodes.attribute_names):
            result.attributes[prop] = nodes.get_attribute(prop, _all)

        for prop in sorted(nodes.dynamics_attribute_names):
            result.dynamics_attributes[prop] = nodes.get_dynamics_attribute(prop, _all)

        return result

    def save(self, filepath):
        """
        Save to SONATA Nodes HDF5.

        TODO: move to `libsonata`?
        """
        def _write_group(data, out):
            for prop, column in data.iteritems():
                values = column.values
                if values.dtype == np.object:
                    dt = h5py.special_dtype(vlen=six.text_type)
                    out.create_dataset(prop, data=values, dtype=dt)
                else:
                    out.create_dataset(prop, data=values)

        with h5py.File(filepath, 'w') as h5f:
            root = h5f.create_group('/nodes/%s' % self.name)
            root.create_dataset('node_type_id', data=np.full(self.size, -1))
            _write_group(self._attributes, root.create_group('0'))
            _write_group(self._dynamics_attributes, root.create_group('0/dynamics_params'))
