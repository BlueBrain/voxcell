"""Cell collection access / writer."""
import collections

import h5py
import numpy as np
import pandas as pd

from voxcell.exceptions import VoxcellError
from voxcell.math_utils import angles_to_matrices, euler2mat, mat2euler
from voxcell.quaternion import matrices_to_quaternions, quaternions_to_matrices


def _load_sonata_orientations(group, cells):
    """Set CellCollection rotation matrices from a sonata file."""
    dataset_names = [name for name in list(group) if isinstance(group[name], h5py.Dataset)]
    if not dataset_names:
        return None
    size = group[dataset_names[0]].shape[0]

    def _get_values(properties):
        """Retrieve prop from the group if exists."""
        return [group.get(prop, np.zeros((size,))) for prop in properties]

    props = np.array([
        'orientation_x',
        'orientation_y',
        'orientation_z',
        'orientation_w',
    ])
    orientation_count = np.count_nonzero(np.isin(props, list(group)))
    if orientation_count == 4:
        cells.orientation_format = "quaternions"
        cells.orientations = quaternions_to_matrices(np.vstack(_get_values(props)).T)
    elif orientation_count in [1, 2, 3]:
        raise VoxcellError(
            "Missing orientation fields. Should be 4 quaternions or some euler angles or nothing")
    else:
        # need to keep this rotation_angle ordering for euler2mat (expects z, y, x)
        props = np.array([
            'rotation_angle_zaxis',
            'rotation_angle_yaxis',
            'rotation_angle_xaxis',
        ])
        cells.orientation_format = "eulers"
        cells.orientations = euler2mat(*_get_values(props))


def _load_property(properties, name, values, library_group=None):
    """Loads single property with respect to a library group if presented.

    Args:
        properties (pd.DataFrame): properties
        name (str): property name
        values (array-like): property values
        library_group (h5py.Group): library group
    """
    if values.dtype == object:
        values = values.asstr()
    values = values[()]
    if library_group is not None and name in library_group:
        if library_group[name].dtype == object:
            unique_values = library_group[name].asstr()[()]
        else:
            unique_values = library_group[name][()]

        if unique_values.size < 0.5 * values.size:
            properties[name] = pd.Categorical.from_codes(values, categories=unique_values)
        else:
            properties[name] = unique_values[values]
    else:
        properties[name] = values


def _is_string_enum(series):
    """Whether ``series`` contains enum of strings."""
    is_cat_str = (isinstance(series.dtype, pd.CategoricalDtype) and
                  series.dtype.categories.dtype == object)
    return series.dtype == object or is_cat_str


class CellCollection:
    """Encapsulates all the data related to a collection of cells that compose a circuit.

    Multi-dimensional properties (such as positions and orientations) are attributes.
    General properties are a in a pandas DataFrame object "properties".
    """

    # properties that start with it are dynamic, and handled appropriately, see `dynamics_params` in
    # https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-nodes
    SONATA_DYNAMIC_PROPERTY = '@dynamics:'

    def __init__(self, population_name='default', orientation_format="quaternions"):
        """Init CellCollection.

        Args:
            population_name: SONATA population name, currently assume a single population collection
            orientation_format: quaternions or eulers.
        """
        self.population_name = population_name
        self.positions = None
        self.orientations = None
        self.properties = pd.DataFrame()
        self._orientation_format = orientation_format

    def _nonzero_sizes(self):
        return list({len(obj) for obj in [self.properties, self.positions, self.orientations] if
                     obj is not None and len(obj) != 0})

    def _check_sizes(self):
        if len(self._nonzero_sizes()) > 1:
            raise VoxcellError("Lengths of properties, position, orientation don't match")

    def size(self):
        """Return the size of the CellCollection.

        If the positions, orientations or properties are filled with values it checks the sizes of
        these objects and then return the size of the CellCollection.
        """
        self._check_sizes()
        sizes = self._nonzero_sizes()
        return sizes[0] if sizes else 0

    def __len__(self):
        """Return the length of the CellCollection."""
        return self.size()

    @property
    def orientation_format(self):
        """Return the format of the orientation either "eulers" or "quaternions"."""
        return self._orientation_format

    @orientation_format.setter
    def orientation_format(self, val):
        """Set the format of orientation with only "eulers" or "quaternions"."""
        if val not in ["quaternions", "eulers"]:
            raise VoxcellError('You must set orientation_type to either "quaternions" or "eulers".')
        self._orientation_format = val

    def add_properties(self, new_properties, overwrite=True):
        """Adds new columns to the properties DataFrame.

        Args:
            new_properties: a pandas DataFrame object
            overwrite: if True, overwrites columns with the same name.
            Otherwise, a VoxcellError is raised.
        """
        for name, prop in new_properties.items():
            if (not overwrite) and (name in self.properties):
                raise VoxcellError(f"Column '{name}' already exists")
            self.properties[name] = prop

    def remove_unassigned_cells(self):
        """Remove cells with one or more unassigned property."""
        idx_unassigned = self.properties[self.properties.isnull().any(axis=1)].index
        self.properties = self.properties.drop(idx_unassigned)
        self.properties.reset_index(inplace=True, drop=True)
        if self.orientations is not None:
            self.orientations = np.delete(self.orientations, idx_unassigned, 0)
        if self.positions is not None:
            self.positions = np.delete(self.positions, idx_unassigned, 0)

    def as_dataframe(self, index_offset=1):
        """Return a dataframe with all cell properties.

        Args:
            index_offset: index offset (0 or 1). The default may change to 0 in a future version.
        """
        result = self.properties.copy()
        if self.positions is not None:
            result['x'] = self.positions[:, 0]
            result['y'] = self.positions[:, 1]
            result['z'] = self.positions[:, 2]
        if self.orientations is not None:
            result['orientation'] = list(self.orientations)

        result.index = np.arange(index_offset, len(result) + index_offset)

        result.columns = map(str, result.columns)

        return result

    @classmethod
    def from_dataframe(cls, df, index_offset=1):
        """Return a CellCollection object from a dataframe of cell properties.

        Args:
            df: Pandas DataFrame containing the cell properties, with index starting from 0 or 1.
            index_offset: index offset (0 or 1). The default may change to 0 in a future version.
        """
        if not (df.index == np.arange(index_offset, len(df) + index_offset)).all():
            raise VoxcellError(
                f"Index != {index_offset}..{len(df) + index_offset - 1} (got: {df.index.values})"
            )
        result = cls()
        if 'x' in df:
            result.positions = df[['x', 'y', 'z']].values
        if 'orientation' in df:
            result.orientations = np.stack(df['orientation'])
        # don't use `set` for filtering because it looses the order of columns => the restored Cells
        # will have different columns order which is bad.
        props = [column for column in df.columns if column not in ['x', 'y', 'z', 'orientation']]
        result.properties = df[props].reset_index(drop=True)
        return result

    def save(self, filename):
        """Saves this cell collection to HDF5 file in MVD3 or SONATA format.

        Args:
            filename: filepath to write. If it ends with '.mvd3' then it is treated as MVD3,
                otherwise as SONATA.
        """
        none_properties = self.properties.isnull().any(axis=0)
        if none_properties.any():
            names = none_properties.index[none_properties].to_list()
            raise VoxcellError(f"Replace `None` in {names} properties before saving")
        if str(filename).lower().endswith('mvd3'):
            self.save_mvd3(filename)
        else:
            self.save_sonata(filename)

    def save_mvd3(self, filename):
        """Save this cell collection to mvd3 HDF5.

        Args:
            filename(str): fullpath to filename to write
        """
        self._check_sizes()
        with h5py.File(filename, 'w') as f:
            f.create_group('cells')
            f.create_group('library')

            if self.positions is not None:
                f.create_dataset('cells/positions', data=self.positions)

            if self.orientations is not None:
                f.create_dataset('cells/orientations',
                                 data=matrices_to_quaternions(self.orientations))
            # numpy's `np.object` type must be represented as `str_dt`
            # http://docs.h5py.org/en/latest/strings.html
            str_dt = h5py.special_dtype(vlen=str)
            for name, series in self.properties.items():
                values = series.to_numpy()
                if _is_string_enum(series) and not name.startswith(self.SONATA_DYNAMIC_PROPERTY):
                    unique_values, indices = np.unique(values, return_inverse=True)
                    f.create_dataset('cells/properties/' + name, data=indices.astype(np.uint32))
                    f.create_dataset('library/' + name, data=unique_values, dtype=str_dt)
                else:
                    dt = str_dt if values.dtype == object else values.dtype
                    f.create_dataset('cells/properties/' + name, data=values, dtype=dt)

    @classmethod
    def load(cls, filename):
        """Loads CellCollection from a file.

        Args:
            filename: filepath to cells file. If it ends with '.mvd3' then it is treated as
                MVD3 circuit, otherwise as SONATA.

        Returns:
            CellCollection: loaded cells
        """
        if str(filename).lower().endswith('mvd3'):
            return cls.load_mvd3(filename)
        return cls.load_sonata(filename)

    @classmethod
    def load_mvd2(cls, filename):
        """Load a cell collection from mvd2 HDF5.

        This method is a copy of `loadMVD2` from bluepy/v2/impl/cells_mvd.py

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        """

        def parse_neuron_line(line):
            """Parser for neurons."""
            tokens = line.split()
            return {
                'morphology': tokens[0],
                'hypercolumn': int(tokens[2]),
                'minicolumn': int(tokens[3]),
                'layer': 1 + int(tokens[4]),
                'mtype': mtypes[int(tokens[5])],
                'morph_class': morph_classes[int(tokens[5])],
                'synapse_class': synapse_classes[int(tokens[5])],
                'etype': etypes[int(tokens[6])],
                'x': float(tokens[7]),
                'y': float(tokens[8]),
                'z': float(tokens[9]),
                'orientation': float(tokens[10]),
                'me_combo': tokens[11]
            }

        SECTIONS = (
            "Neurons Loaded", "MicroBox Data",
            "MiniColumnsPosition", "CircuitSeeds",
            "MorphTypes", "ElectroTypes", "FOOTER"
        )

        CATEGORICAL_PROPS = [
            'mtype', 'etype', 'morph_class', 'synapse_class',
            'morphology', 'me_combo'
        ]

        section_lines = collections.defaultdict(list)
        accumulator = None

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line in SECTIONS:
                    accumulator = section_lines[line]
                elif accumulator is not None:
                    accumulator.append(line)

        mtypes = [line.split()[0] for line in section_lines['MorphTypes']]
        morph_classes = [line.split()[1] for line in section_lines['MorphTypes']]
        synapse_classes = [line.split()[2] for line in section_lines['MorphTypes']]
        etypes = [line.split()[0] for line in section_lines['ElectroTypes']]

        result = pd.DataFrame(list(map(parse_neuron_line, section_lines['Neurons Loaded'])))
        for c in CATEGORICAL_PROPS:
            result[c] = result[c].astype('category')

        result['orientation'] = list(angles_to_matrices(np.pi * result['orientation'] / 180, 'y'))

        result.index = 1 + np.arange(len(result))
        return cls.from_dataframe(result)

    @classmethod
    def load_mvd3(cls, filename):
        """Load a cell collection from mvd3 HDF5.

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        """
        cells = cls()

        with h5py.File(filename, 'r') as f:
            data = f['cells']
            if 'positions' in data:
                cells.positions = np.array(data['positions'])

            if 'orientations' in data:
                cells.orientations = np.array(data['orientations'])
                cells.orientations = quaternions_to_matrices(cells.orientations)

            if 'properties' in data:
                for name, values in data['properties'].items():
                    _load_property(cells.properties, name, values, f.get('library'))
        return cells

    def save_sonata(self, filename, forced_library=None, mode='w'):
        """Save this cell collection to sonata HDF5.

        Args:
            filename(str): fullpath to filename to write
            forced_library(iterable of str): names of properties that are
            forced to become part of the @library
            mode(str): mode used to create/open file; passed directly to h5py.File:
                'w' overwrites, 'a' appends

        Note:
          * Only properties that contain strings can be included in the @library
          * when forced_library is None, properties that are categorical are
            included in the @library, unless their number of unique values is
            more than half of all the values
        """
        # pylint: disable=too-many-locals
        forced_library = set() if forced_library is None else set(forced_library)

        self._check_sizes()
        with h5py.File(filename, mode) as h5f:
            population = h5f.create_group(f'/nodes/{self.population_name}')
            population.create_dataset('node_type_id', data=np.full(len(self.properties), -1))
            group = population.create_group('0')
            str_dt = h5py.special_dtype(vlen=str)
            for name, series in self.properties.items():
                if name.startswith(self.SONATA_DYNAMIC_PROPERTY):
                    name = name.split(self.SONATA_DYNAMIC_PROPERTY)[1]
                    dt = str_dt if series.dtype == object else series.dtype
                    group.create_dataset(
                        f'dynamics_params/{name}',
                        data=series.to_numpy(),
                        dtype=dt,
                    )
                elif _is_string_enum(series) or (series.dtype == object and name in forced_library):
                    indices, unique_values = series.factorize()
                    if name in forced_library or len(unique_values) < .5 * len(indices):
                        group.create_dataset(name, data=indices.astype(np.uint32))
                        group.create_dataset(f'@library/{name}', data=unique_values, dtype=str_dt)
                    else:
                        group.create_dataset(name, data=series.to_numpy(), dtype=str_dt)
                else:
                    group.create_dataset(name, data=series.to_numpy())

            if self.orientations is not None:
                if self.orientation_format == "quaternions":
                    quaternions = matrices_to_quaternions(self.orientations)
                    group.create_dataset('orientation_x', data=quaternions[:, 0])
                    group.create_dataset('orientation_y', data=quaternions[:, 1])
                    group.create_dataset('orientation_z', data=quaternions[:, 2])
                    group.create_dataset('orientation_w', data=quaternions[:, 3])
                elif self.orientation_format == "eulers":
                    az, ay, ax = mat2euler(self.orientations)
                    group.create_dataset('rotation_angle_xaxis', data=ax)
                    group.create_dataset('rotation_angle_yaxis', data=ay)
                    group.create_dataset('rotation_angle_zaxis', data=az)

            if self.positions is not None:
                group.create_dataset('x', data=self.positions[:, 0])
                group.create_dataset('y', data=self.positions[:, 1])
                group.create_dataset('z', data=self.positions[:, 2])

    @classmethod
    def load_sonata(cls, filename, population_name=None):
        """Loads a cell collection from sonata HDF5.

        Args:
            filename(str): fullpath to filename to read

        Returns:
            CellCollection object
        """
        cells = cls()

        with h5py.File(filename, 'r') as h5f:
            if population_name is None:
                population_names = list(h5f['/nodes'].keys())
                assert len(population_names) == 1, 'Single population is supported only'
                population_name = population_names[0]

            population = h5f[f'/nodes/{population_name}']
            cells.population_name = population_name

            assert '0' in population, 'Single group "0" is supported only'
            group = population['0']

            keys = set(group.keys())
            if 'x' in group:
                cells.positions = np.vstack((group['x'], group['y'], group['z'])).T

            rotation_datasets = {'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w',
                                 'rotation_angle_zaxis', 'rotation_angle_yaxis',
                                 'rotation_angle_xaxis'}
            if len(rotation_datasets - keys) != len(rotation_datasets):
                _load_sonata_orientations(group, cells)
            properties_names = keys - {'x', 'y', 'z'}.union(rotation_datasets)
            for name in properties_names:
                if not isinstance(group[name], h5py.Dataset):
                    continue
                _load_property(cells.properties, name, group[name], group.get('@library'))

            if 'dynamics_params' in group:
                for name, values in group['dynamics_params'].items():
                    if not isinstance(values, h5py.Dataset):
                        continue
                    if values.dtype == object:
                        values = values.asstr()
                    cells.properties[cls.SONATA_DYNAMIC_PROPERTY + name] = values[()]

        return cells

    def __str__(self):
        """Return the string describing the CellCollection."""
        properties = list(self.properties.columns)

        if self.positions is not None:
            properties += list('xyz')

        if self.orientations is not None:
            if self.orientation_format == "quaternions":
                properties += ['orientation_x',
                               'orientation_y',
                               'orientation_z',
                               'orientation_w',
                               ]
            elif self.orientation_format == "eulers":
                properties += ['rotation_angle_xaxis',
                               'rotation_angle_yaxis',
                               'rotation_angle_zaxis',
                               ]

        return (f'CellCollection[population_name: {self.population_name}]: '
                f'properties: {properties} '
                f'orientation_format: {self.orientation_format} '
                f'count: {len(self.properties)}'
                )

    __repr__ = __str__
