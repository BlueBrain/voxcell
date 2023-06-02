"""Library to handle traits fields and collections and the logic to assign them.

A "trait" is a set of properties and their values.
A "traits collection" is a group of traits. It's represented as a pandas
DataFrame object where each trait is a row and each property is a column.
For example::

       sclass      mtype
    0  INH         Pyramidal
    1  EXC         Pyramidal
    2  INH         Martinotti

A "probability distribution" specifies the probability of each trait in a collection
being assigned to a given cell.
A "distributions collection" is a group of distributions. It's represented
as a pandas DataFrame object where each trait is a row and each distribution
is a column.
For example::

        0     1    2     3
     0  0.25  0.5  0.5   0.0
     1  0.75  0.5  0.25  0.1
     2  0.0   0.0  0.25  0.9

The distribution number 3 in the above table indicates 10% chances of picking
the trait 1 (an excitatory Pyramidal cell) and 90% of picking trait 2
(an inhibitory Martinotti cell).
"""

import logging

import numpy as np
import pandas as pd

L = logging.getLogger(__name__)


class SpatialDistribution:
    """Encapsulates the data relative to the 3D probability distributions of traits.

    Args:
        field: VoxelData where every voxel is an index in the distributions list
               -1 means unavailable data.
        distributions: a distributions collection, see module docstring
        traits: a traits collection, see module docstring
    """

    def __init__(self, field, distributions, traits):
        """Init SpatialDistribution."""
        self.field = field
        self.distributions = distributions / distributions.sum()
        self.traits = traits

    def assign(self, positions):
        """For every cell in positions, chooses a property from a spatial distribution.

        Args:
            positions: list of positions for soma centers (x, y, z).

        Returns:
            An array with the same length as positions where each value
            is an index into spatial_dist.traits
        """
        dist_id_per_position = self.field.lookup(positions)

        unknown_count = np.count_nonzero(dist_id_per_position == -1)
        if unknown_count:
            L.warning('%d total positions in unknown areas', unknown_count)

        valid = np.in1d(dist_id_per_position, self.distributions.columns)
        if np.count_nonzero(~valid):
            L.warning('missing distribution for %d positions', np.count_nonzero(~valid))

        chosen_trait_indices = np.ones_like(dist_id_per_position, dtype=np.int32) * -1

        unique_dists = np.unique(dist_id_per_position[valid])
        for dist_id, dist in self.distributions[unique_dists].items():
            hit_count = np.count_nonzero(dist_id_per_position == dist_id)
            chosen = np.random.choice(dist.index, hit_count, p=dist.values)
            chosen_trait_indices[dist_id_per_position == dist_id] = chosen

        return chosen_trait_indices

    def assign_conditional(self, positions, preassigned):
        """For every cell in positions, choose a property from a spatial distribution.

        For every cell in positions, choose a property from a spatial distribution,
        but taking into account a pre-assigned values of related properties.

        Args:
            positions: list of positions for soma centers (x, y, z).
            preassigned: pandas.DataFrame or pandas.Series with the pre-assigned values.

        Returns:
            An array with the same length as positions where each value
            is an index into spatial_dist.traits

            For those positions whose value could not be determined, -1 is used.
        """
        # No need to apply offset here because it's applied inside subdist.assign
        preassigned = preassigned.to_frame() if isinstance(preassigned, pd.Series) else preassigned

        subsections = self.split(list(preassigned.columns))
        chosen = np.ones(shape=(len(preassigned)), dtype=int) * -1

        unique_assigned = preassigned.drop_duplicates()
        for values_comb in unique_assigned.values:
            hashable = tuple(values_comb)

            if hashable in subsections:
                subdist = subsections[hashable]
                mask = np.all(np.array(preassigned) == values_comb, axis=1)
                chosen[mask] = subdist.assign(positions[mask])

        invalid_count = np.count_nonzero(chosen == -1)
        if invalid_count:
            # this may happen because of inconsistencies of the data
            # for example if we assigned excitatory neurons to a neuron that is in
            # a voxel for which only inhibitory metype probabilities are known
            L.warning('%d / %d = %f cells could not get a valid value assigned',
                      invalid_count, len(chosen),
                      float(invalid_count) / len(chosen))

        return chosen

    def collect(self, positions, preassigned=None, names=None):
        """For every cell in positions, chose trait values from a spatial distribution.

        For every cell in positions, chose trait values from a spatial distribution,
        taking into account a pre-assigned values of related properties (if specified).

        Args:
            positions: list of positions for soma centers (x, y, z).
            preassigned: pandas.DataFrame or pandas.Series with the pre-assigned values.
            names: names of the properties to collect. If not specified use all in traits.

        Returns:
            A pandas DataFrame with one row for each index and one column for each value of names
        """
        if preassigned is None:
            return self.collect_traits(self.assign(positions), names)

        return self.collect_traits(self.assign_conditional(positions, preassigned), names)

    def collect_traits(self, chosen, names=None):
        """Return the trait values corresponding to an array of indices.

        Args:
            chosen: array of indices into the traits dataframe
            names: names of the properties to collect. If not specified use all in traits.

        Returns:
            A pandas DataFrame with one row for each index and one column for each value of names
        """
        names = names if names is not None else self.traits.columns
        df = self.traits[names].iloc[chosen]
        return df.reset_index().drop('index', axis=1)

    def split(self, attributes):
        """Split a distribution in two or more.

        Split a distribution in two or more so that each one only references
        traits with the same value for certain attributes.
        Each resulting distribution is renormalised.

        this may be generating distributions that are empty but that should not be a problem

        Note that because for every distribution we are creating a new one,
        the indexes of any associated field are still valid.

        Returns:
            A dictionary where the keys are tuples with the values of the attributes found
            in the traits_collection and the values are the resulting SpatialDistribution objects.
        """
        grouped_distributions = {}
        for attr_values, traits in self.traits.groupby(attributes):
            # this is backwards compatibility: https://github.com/pandas-dev/pandas/issues/42795
            if len(attributes) == 1:
                attr_values = tuple(attr_values)

            dists = self.distributions.iloc[traits.index]

            # remove dists that have become empty
            dists = dists[dists.columns[dists.sum() != 0]]
            dists /= dists.sum()

            grouped_distributions[attr_values] = SpatialDistribution(self.field, dists, traits)

        return grouped_distributions

    def reduce(self, attribute):
        """Return a SpatialDistribution with only the given attribute.

        Given a spatial distribution, extract an equivalent one where all of the properties
        of the traits collection have been removed except for a specific one.

        For example, taking the traits collection::

               sclass      mtype
            0  INH         Pyramidal
            1  EXC         Pyramidal
            2  INH         Martinotti

        and the distributions collection::

               0
            0  0.2
            1  0.4
            2  0.4

        Ignoring all properties except 'sclass' would give the simplified traits collection::

               sclass
            0  INH
            1  EXC

        and the distributions collection::

               0
            0  0.6
            1  0.4

        Note that because for every distribution we are creating a new one,
        the indexes of any associated field are still valid.

        Returns:
            An SpatialDistribution object
        """
        traits, distributions = [], []
        for value, data in self.traits.groupby(attribute):
            traits.append(value)
            distributions.append(self.distributions.iloc[data.index].sum())

        traits = pd.DataFrame(traits, columns=[attribute])
        distributions = pd.DataFrame(
            distributions,
            columns=self.distributions.columns,
            index=traits.index
        )

        return SpatialDistribution(self.field, distributions, traits)

    def get_probability_field(self, attribute, value):
        """Extract the probability of a particular attribute value from a spatial distribution.

        Voxels where the probability is missing will contain a probability value of -1.

        Returns:
            A VoxelData with the same shape as sdist.field where every voxel is a float
            representing the probability of an attribute value being assigned to a cell
            in that voxel. For voxels with unknown values -1 is used.
        """
        probs = self.distributions[self.traits[attribute] == value].sum()
        probs_field = probs.reindex(self.field.raw.flatten())
        probs_field = probs_field.fillna(-1)
        return self.field.with_data(probs_field.values.reshape(self.field.raw.shape))

    @classmethod
    def from_probability_field(cls, probs_field, name, positive_value, negative_value):
        """Creates a binary SpatialDistribution object from a single probability field.

        Args:
            probs_field: A VoxelData where every voxel is a float representing the probability
                of an attribute value being assigned to a cell in that voxel. For voxels with
                unknown values -1 is used.
            name: name of the trait.
            positive_value: value of the trait that corresponds to probs_field.
            negative_value: value of the trait that corresponds to the probability field
                complementary to probs_field.

        Returns:
            A SpatialDistribution object
        """
        valid = probs_field.raw != -1

        probs_field_complete = probs_field.raw.copy()
        any_value = probs_field.raw[valid].flatten()[0]
        probs_field_complete[~valid] = any_value
        unique_probs, field = np.unique(probs_field_complete, return_inverse=True)

        field = field.reshape(probs_field.raw.shape)
        field[~valid] = -1
        field = probs_field.with_data(field)

        distributions = pd.DataFrame([unique_probs, 1 - unique_probs])
        traits = pd.DataFrame({name: [positive_value, negative_value]})

        return cls(field, distributions, traits)

    def drop_duplicates(self, decimals=None):
        """Return a new SpatialDistribution with duplicate distributions removed.

        Args:
            decimals: Number of decimal places to round each column to beforehand.
        """
        dists, inverse = _drop_duplicate_columns(self.distributions, decimals)

        field = np.ones_like(self.field.raw) * -1
        for col_idx, unique_idx in enumerate(inverse):
            field[self.field.raw == col_idx] = unique_idx

        return SpatialDistribution(self.field.with_data(field), dists, self.traits.copy())


def _drop_duplicate_columns(dataframe, decimals=None):
    """Drop duplicate columns from a dataframe and return the indices of the columns.

    Drop duplicate columns from a dataframe and return the indices of the columns
    of the resulting dataframe that can be used to reconstruct the original one.

    Args:
        dataframe: pandas.Dataframe object
        decimals: Number of decimal places to round each column to beforehand.
    """
    df = dataframe.transpose()

    if decimals is not None:
        df = df.round(decimals)

    g = df.groupby(list(df.columns))
    df1 = df.set_index(list(df.columns))
    tags = df1.index.map(lambda ind: g.indices[ind][0])
    unique_rows_idx, inverse = np.unique(tags, return_inverse=True)

    return df.iloc[unique_rows_idx].transpose(), inverse
