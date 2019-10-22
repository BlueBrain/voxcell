"""
SONATA Nodes access / writer.

A thin wrapper around `libsonata` with some syntactic sugar and Pandas view.

See also:
        https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#neuron_networks_nodes

TODO:
    Use `libsonata` instead, once it provides *write* functionality (?)
"""

import logging

import h5py
import numpy as np
import pandas as pd
import six

from libsonata import NodeStorage, Selection

from voxcell.exceptions import VoxcellError
from voxcell.math_utils import euler2mat, mat2euler


L = logging.getLogger(__name__)


def _load_mecombo_info(filepath):
    '''load mecombo information

    Note: this is sometimes known as the '.tsv file', or MEComboInfoFile
    '''
    def usecols(name):
        '''pick the needed columns'''
        return name not in ('morph_name', 'layer', 'fullmtype', 'etype')

    return pd.read_csv(filepath, sep=r'\s+', usecols=usecols, index_col='combo_name')


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
    # pylint: disable=abstract-method
    # newer pylints complain about the missing _constructor_expanddim property
    # being abstract, but I don't think it makes sense to have it - I could be wrong,
    # if people get 'method not implemented', then we'll have to add it.

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
        """ Convert into pandas DataFrame.

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
        """Load NodePopulation from SONATA Nodes HDF5.

        Limitations:
          - no CSV support
          - one population per file
          - one group per population
        """
        nodes = _open_population(filepath)

        result = cls(nodes.name, nodes.size)

        _all = Selection([(0, nodes.size)])

        for prop in sorted(nodes.attribute_names):
            if prop in nodes.enumeration_names:
                result.attributes[prop] = pd.Categorical.from_codes(
                    nodes.get_enumeration(prop, _all),
                    categories=nodes.enumeration_values(prop))
            else:
                result.attributes[prop] = nodes.get_attribute(prop, _all)

        for prop in sorted(nodes.dynamics_attribute_names):
            result.dynamics_attributes[prop] = nodes.get_dynamics_attribute(prop, _all)

        return result

    @staticmethod
    def _check_props_can_be_in_library(library_properties, attributes):
        missing_props, non_string_props = [], []
        for prop in library_properties:
            if prop not in attributes.columns:
                missing_props.append(prop)
            if attributes[prop].dtype != np.object:
                non_string_props.append(prop)

        if missing_props:
            raise VoxcellError("Properties %s are not in attributes" % missing_props)

        if non_string_props:
            raise VoxcellError("Properties %s are not strings, and can't be in the library" %
                               missing_props)

    @staticmethod
    def _write_string_library(group, name, unique_values, indices):
        dt = h5py.special_dtype(vlen=six.text_type)
        group.create_dataset(name, data=indices.astype(np.uint32))
        group.create_dataset('@library/%s' % name, data=unique_values, dtype=dt)

    def save(self, filepath, library_properties=None):
        """Save to SONATA Nodes HDF5.

        library_properties(list of string): properties that should be converted
        to a '@library' enumeration; see SONATA spec for more info

        TODO: move to `libsonata`?
        """
        library_properties = set() if library_properties is None else set(library_properties)

        dt = h5py.special_dtype(vlen=six.text_type)

        def _write_group(data, out):
            for prop, column in data.iteritems():
                values = column.values
                if values.dtype == np.object:
                    if prop in library_properties:
                        unique_values, indices = np.unique(values, return_inverse=True)
                        self._write_string_library(out, prop, unique_values, indices)
                    else:
                        out.create_dataset(prop, data=values, dtype=dt)
                elif pd.api.types.is_categorical_dtype(values):
                    self._write_string_library(out,
                                               prop,
                                               unique_values=values.categories.values,
                                               indices=values.codes)
                else:
                    out.create_dataset(prop, data=values)

        self._check_props_can_be_in_library(library_properties, self._attributes)

        with h5py.File(filepath, 'w') as h5f:
            root = h5f.create_group('/nodes/%s' % self.name)
            root.create_dataset('node_type_id', data=np.full(self.size, -1))
            _write_group(self._attributes, root.create_group('0'))
            _write_group(self._dynamics_attributes, root.create_group('0/dynamics_params'))

    @classmethod
    def from_cell_collection(cls, cell_collection, population_name, mecombo_info_path=None):
        """ Convert CellCollection to SONATA NodePopulation. """
        if 'me_combo' in cell_collection.properties:
            if mecombo_info_path is None:
                L.warning("Please specify 'mecombo_info_path' in order to "
                          "resolve 'me_combo' property")
            else:
                mecombo_info = _load_mecombo_info(mecombo_info_path)
                L.info("'me_combo' property would be resolved to 'model_template' "
                       "and dynamics parameters %s",
                       ", ".join("'%s'" % s
                                 for s in mecombo_info.columns if s != 'emodel'))

        result = cls(population_name, size=len(cell_collection.properties))

        if cell_collection.positions is not None:
            result.positions = cell_collection.positions

        if cell_collection.orientations is not None:
            result.orientations = cell_collection.orientations

        for prop, column in cell_collection.properties.iteritems():
            if prop == 'me_combo':
                continue
            result.attributes[prop] = column.values

        if 'me_combo' in cell_collection.properties and mecombo_info_path:
            mecombo_params = mecombo_info.loc[cell_collection.properties['me_combo']]
            for prop, column in mecombo_params.iteritems():
                values = column.values
                if prop == 'emodel':
                    values = [('hoc:' + v) for v in values]
                    result.attributes['model_template'] = values
                else:
                    result.dynamics_attributes[prop] = values

        return result
