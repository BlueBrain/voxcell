'''library to handle traits fields and collections and the logic to assign them

    A "trait" is a set of properties and their values.
    A "traits collection" is a group of traits. It's represented as a pandas
    DataFrame object where each trait is a row and each property is a column.
    For example:
           sclass      mtype
        0  inhibitory  Pyramidal
        1  excitatory  Pyramidal
        2  inhibitory  Martinotti

    A "probability distribution" specifies the probability of each trait in a collection
    being assigned to a given cell.
    A "distributions collection" is a group of distributions. It's represented
    as a pandas DataFrame object where each trait is a row and each distribution
    is a column. For example:
            0     1    2     3
         0  0.25  0.5  0.5   0.0
         1  0.75  0.5  0.25  0.1
         2  0.0   0.0  0.25  0.9

    The distribution number 3 in the above table indicates 10% chances of picking
    the trait 1 (an excitatory Pyramidal cell) and 90% of picking trait 2
    (an inhibitory Martinotti cell).
'''

import numpy as np
import pandas as pd


import logging
L = logging.getLogger(__name__)


class SpatialDistribution(object):
    '''encapsulates the data relative to the 3D probability distributions of traits'''

    def __init__(self, field, distributions, traits):
        '''
        Args:
            field: VoxelData where every voxel is an index in the distributions list
                   -1 means unavailable data.
            distributions: a distributions collection, see module docstring
            traits: a traits collection, see module docstring
        '''
        self.field = field
        self.distributions = distributions
        self.traits = traits

    def assign(self, positions):
        '''for every cell in positions, chooses a property from a spatial distribution

        Args:
            positions: list of positions for soma centers (x, y, z).

        Returns:
            An array with the same length as positions where each value
            is an index into spatial_dist.traits
        '''
        dist_id_per_position = self.field.lookup(positions)

        unknown_count = np.count_nonzero(dist_id_per_position == -1)
        if unknown_count:
            L.warning('%d total positions in unknown areas', unknown_count)

        valid = np.in1d(dist_id_per_position, self.distributions.columns)
        if np.count_nonzero(~valid):
            L.warning('missing distribution for %d positions', np.count_nonzero(~valid))

        chosen_trait_indices = np.ones_like(dist_id_per_position) * -1

        unique_dists = np.unique(dist_id_per_position[valid])
        for dist_id, dist in self.distributions[unique_dists].iteritems():
            hit_count = np.count_nonzero(dist_id_per_position == dist_id)

            chosen = np.random.choice(dist.keys(), hit_count, p=dist.values)
            chosen_trait_indices[dist_id_per_position == dist_id] = chosen

        return chosen_trait_indices

    def assign_conditional(self, positions, preassigned):
        '''for every cell in positions, chooses a property from a spatial distribution
        but taking into account a pre-assigned values of related properties.


        Args:
            positions: list of positions for soma centers (x, y, z).
            preassigned: pandas.DataFrame or pandas.Series with the pre-assigned values.

        Returns:
            An array with the same length as positions where each value
            is an index into spatial_dist.traits

            For those positions whose value could not be determined, -1 is used.
        '''
        # No need to apply offset here because it's applied inside subdist.assign

        preassigned = preassigned.to_frame() if isinstance(preassigned, pd.Series) else preassigned

        subsections = self.split(tuple(preassigned.columns))
        chosen = np.ones(shape=(len(preassigned)), dtype=np.int) * -1

        unique_assigned = preassigned.drop_duplicates()
        for values_comb in unique_assigned.values:

            if len(values_comb) == 1:
                values_comb = values_comb[0]
                hashable = values_comb
            else:
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

    def collect_traits(self, chosen, names=None):
        '''return the trait values corresponding to an array of indices

        Args:
            chosen: array of indices into the traits dataframe
            names: names of the properties to collect. If not specified use all in traits.

        Returns:
            A pandas DataFrame with one row for each index and one column for each value of names
        '''
        names = names if names is not None else self.traits.columns
        df = self.traits[names].ix[chosen]
        return df.reset_index().drop('index', 1)

    def split(self, attributes):
        '''split a distribution in two or more so that each one only references
        traits with the same value for certain attributes.
        Each resulting distribution is renormalised.

        this may be generating distributions that are empty but that should not be a problem

        Note that because for every distribution we are creating a new one,
        the indexes of any associated field are still valid.

        Returns:
            A dictionary where the keys are tuples with the values of the attributes found
            in the traits_collection and the values are the resulting SpatialDistribution objects.
        '''
        grouped_distributions = {}
        for attr_values, traits in self.traits.groupby(attributes):

            dists = self.distributions.ix[traits.index]

            # remove dists that have become empty
            dists = dists[dists.columns[dists.sum() != 0]]
            dists /= dists.sum()

            grouped_distributions[attr_values] = SpatialDistribution(self.field, dists, traits)

        return grouped_distributions

    def reduce(self, attribute):
        '''given a spatial distribution, extract an equivalent one where all of the properties
        of the traits collection have been removed except for a specific one.

        For example:

            Taking the traits collection:
                   sclass      mtype
                0  inhibitory  Pyramidal
                1  excitatory  Pyramidal
                2  inhibitory  Martinotti

            And the distributions collection:
                   0
                0  0.2
                1  0.4
                2  0.4

            Ignoring all properties except 'sclass' would give the simplified traits collection:
                   sclass
                0  inhibitory
                1  excitatory

            And the distributions collection:
                   0
                0  0.6
                1  0.4

        Note that because for every distribution we are creating a new one,
        the indexes of any associated field are still valid.

        Returns:
            An SpatialDistribution object
        '''

        distributions = dict((value, self.distributions.ix[data.index].sum())
                             for value, data in self.traits.groupby(attribute))

        traits = pd.DataFrame(distributions.keys(), columns=[attribute])
        distributions = pd.DataFrame(distributions.values(),
                                     columns=self.distributions.columns,
                                     index=traits.index)

        return SpatialDistribution(self.field,
                                   distributions,
                                   traits)

    def get_probability_field(self, attribute, value):
        '''extract the probability of a particular attribute value from a spatial distribution

        Voxels where the probability is missing will contain a probability of zero.

        Returns:
            A numpy array with the same shape as sdist.field where every voxel is a float
            representing the probability of an attribute value being assigned to a cell
            in that voxel.
        '''
        probs = self.distributions[self.traits[attribute] == value].sum()
        probs_field = probs[self.field.flatten()]
        probs_field = probs_field.fillna(0)
        return probs_field.values.reshape(self.field.shape)
