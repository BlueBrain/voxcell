from nose.tools import eq_

from voxcell import core
from voxcell.traits import SpatialDistribution, _drop_duplicate_columns
import pandas as pd
import numpy as np
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal


NOFIELD = np.array([])


def test_split_distribution_collection_empty_0():
    probabilities = pd.DataFrame()
    attributes = ('name',)
    traits = pd.DataFrame(columns=attributes)

    eq_(SpatialDistribution(NOFIELD, probabilities, traits).split(attributes),
        {})


def test_split_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, distributions)
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_empty_2():
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame(index=[0]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame(index=[1]))
    # {('a',): SD(NOFIELD, [], traits), ('b',): SD(NOFIELD, [], traits)}


def test_split_distribution_collection_single_0():
    distributions = pd.DataFrame([{0: 1}])
    traits = pd.DataFrame([{'name': 'a'}])
    attributes = ('name',)

    # {('a',): SD(NOFIELD, [{0: 1}], traits)})

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_single_1():
    distributions = pd.DataFrame([0.25, 0.75])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    attributes = ('name',)

    # {('a',): SD(NOFIELD, [{0: 1.0}], traits), ('b',): SD(NOFIELD, [{1: 1.0}], traits)})
    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame([1.], index=[1]))


def test_split_distribution_collection_single_2():
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'b'}])
    attributes = ('name',)

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame([0.5, 0.5], index=[1, 2]))


def test_split_distribution_collection_multiattr_0():
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'y'},
                           {'name': 'b', 'type': 'y'}])

    attributes = ('name', 'type')

    # {('a', 'x'): SD(NOFIELD, [{0: 1.0}], traits),
    #  ('b', 'y'): SD(NOFIELD, [{1: 0.5, 2: 0.5}], traits)})

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.0]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([0.5, 0.5], index=[1, 2]))


def test_split_distribution_collection_multiattr_1():
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'},
                           {'name': 'b', 'type': 'y'}])

    attributes = ('name', 'type')

    res = SpatialDistribution(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y'), ('b', 'x')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res[('b', 'x')].distributions, pd.DataFrame([1.], index=[1]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([1.], index=[2]))

    # {('a', 'x'): SD(NOFIELD, [{0: 1.0}], traits),
    #  ('b', 'x'): SD(NOFIELD, [{1: 1.0}], traits),
    #  ('b', 'y'): SD(NOFIELD, [{2: 1.0}], traits)})


def test_reduce_distribution_collection_empty_0():
    sd = SpatialDistribution(NOFIELD, pd.DataFrame(), pd.DataFrame(columns=['name']))
    r = sd.reduce('name')
    assert_frame_equal(r.distributions, pd.DataFrame())


def test_reduce_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'}])
    dists = pd.DataFrame(index=traits.index)
    r = SpatialDistribution(NOFIELD, dists, traits).reduce('type')
    print r.distributions
    print r.traits
    assert_frame_equal(r.distributions, dists)


def test_reduce_distribution_collection_0():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'}])

    dists = pd.DataFrame([0.25, 0.75])

    r = SpatialDistribution(NOFIELD, dists, traits).reduce('type')
    assert_frame_equal(r.distributions,
                       pd.DataFrame([1.]))


def test_reduce_distribution_collection_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                          {'name': 'c', 'type': 'y'}])

    dists = pd.DataFrame([0.75, 0.25])

    r = SpatialDistribution(NOFIELD, dists, traits).reduce('type')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'type': ['y', 'x']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.25, 0.75]))


def test_reduce_distribution_collection_2():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x', 'color': 0},
                           {'name': 'b', 'type': 'x', 'color': 1},
                           {'name': 'c', 'type': 'y', 'color': 2},
                           {'name': 'd', 'type': 'y', 'color': 3}])

    dists = pd.DataFrame([0.1, 0.2, 0.3, 0.4])

    r = SpatialDistribution(NOFIELD, dists, traits).reduce('type')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'type': ['y', 'x']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.7, 0.3]))

    r = SpatialDistribution(NOFIELD, dists, traits).reduce('name')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'name': ['a', 'c', 'b', 'd']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.1, 0.3, 0.2, 0.4]))


def check_assign_conditional(preassigned, expected):
    sdist = SpatialDistribution(field=core.VoxelData(np.array([[[0]]]), (25, 25, 25)),
                                distributions=pd.DataFrame([0.75, 0.25, 0.0]),
                                traits=pd.DataFrame([{'name': 'a', 'type': 'x', 'color': 0},
                                                     {'name': 'b', 'type': 'y', 'color': 0},
                                                     {'name': 'c', 'type': 'z', 'color': 1}]))

    chosen = sdist.assign_conditional(np.array([[1, 1, 1]]), preassigned)

    assert_equal(chosen, np.array([expected]))


def test_assign_conditional_single_series():
    check_assign_conditional(pd.DataFrame({'type': ['x']}).type, 0)
    check_assign_conditional(pd.DataFrame({'type': ['y']}).type, 1)


def test_assign_conditional_single_dataframe():
    check_assign_conditional(pd.DataFrame({'type': ['x']}), 0)
    check_assign_conditional(pd.DataFrame({'type': ['y']}), 1)


def test_assign_conditional_multiple():
    check_assign_conditional(pd.DataFrame({'type': ['x'], 'color': [0]}), 0)
    check_assign_conditional(pd.DataFrame({'type': ['y'], 'color': [0]}), 1)


def test_assign_conditional_single_unknown():
    preassigned = pd.DataFrame({'type': ['unknown']})
    check_assign_conditional(preassigned, -1)


def test_assign_conditional_single_impossible():
    preassigned = pd.DataFrame({'type': ['z']})
    check_assign_conditional(preassigned, -1)


def test_assign_conditional_multiple_unknown():
    preassigned = pd.DataFrame({'type': ['unknown'], 'color': [0]})
    check_assign_conditional(preassigned, -1)


def test_assign_conditional_multiple_impossible_0():
    # probability of this combination is zero
    preassigned = pd.DataFrame({'type': ['z'], 'color': [1]})
    check_assign_conditional(preassigned, -1)


def test_assign_conditional_multiple_impossible_1():
    # preassigned values exist but combination does not
    preassigned = pd.DataFrame({'name': ['a'], 'type': ['y']})
    check_assign_conditional(preassigned, -1)


def test_collect():
    sdist = SpatialDistribution(
        field=core.VoxelData(np.array([[[0]]]), (25, 25, 25)),
        distributions=pd.DataFrame([0.75, 0.25, 0.0]),
        traits=pd.DataFrame([
            {'name': 'a', 'type': 'x', 'color': 0},
            {'name': 'b', 'type': 'y', 'color': 0},
            {'name': 'c', 'type': 'z', 'color': 1}
        ])
    )

    positions = np.array([[1, 1, 1]])
    preassigned = pd.DataFrame([{
        'type': 'y'
    }])

    expected = pd.DataFrame([{
        'color': 0
    }])

    result = sdist.collect(positions, preassigned, names=['color'])
    assert_frame_equal(result, expected)


def test_drop_duplicate_columns_0():
    df = pd.DataFrame([[0.4, 0.2],
                       [0.6, 0.8]])

    unique, inverse = _drop_duplicate_columns(df)

    expected = pd.DataFrame([[0.4, 0.2],
                             [0.6, 0.8]])
    assert_frame_equal(unique, expected)

    assert_equal(inverse, np.array([0, 1]))


def test_drop_duplicate_columns_single_0():
    df = pd.DataFrame([[0.4, 0.4], [0.6, 0.6]])

    unique, inverse = _drop_duplicate_columns(df)

    assert_frame_equal(unique, pd.DataFrame([[0.4], [0.6]]))
    assert_equal(inverse, np.array([0, 0]))


def test_drop_duplicate_columns_single_rounded_0():
    df = pd.DataFrame([[0.39999999999999997, 0.40000000000000002], [0.6, 0.6]])

    unique, inverse = _drop_duplicate_columns(df, decimals=100)

    # no change
    assert_frame_equal(unique, pd.DataFrame([[0.39999999999999997, 0.40000000000000002],
                                             [0.6, 0.6]]))
    assert_equal(inverse, np.array([0, 1]))


def test_drop_duplicate_columns_single_rounded_1():
    df = pd.DataFrame([[0.39999999999999997, 0.40000000000000002], [0.6, 0.6]])

    unique, inverse = _drop_duplicate_columns(df, decimals=20)

    # dropped the second column
    assert_frame_equal(unique, pd.DataFrame([[0.4],
                                             [0.6]]))
    assert_equal(inverse, np.array([0, 0]))


def test_drop_duplicate_columns_1():
    df = pd.DataFrame([[0.4, 0.2, 0.2, 0.4, 0.2],
                       [0.6, 0.8, 0.8, 0.6, 0.8]])

    unique, inverse = _drop_duplicate_columns(df)

    expected = pd.DataFrame([[0.4, 0.2],
                             [0.6, 0.8]])
    assert_frame_equal(unique, expected)

    assert_equal(inverse, np.array([0, 1, 1, 0, 1]))


def test_drop_duplicates():

    sd = SpatialDistribution(
        field=core.VoxelData(np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), (1,)),
        distributions=pd.DataFrame([[0.4, 0.2, 0.2, 0.4, 0.2],
                                    [0.6, 0.8, 0.8, 0.6, 0.8]]),
        traits=pd.DataFrame({'size': ['Large', 'Small'], 'speed': ['Fast', 'Slow']}))

    result = sd.drop_duplicates()

    assert_equal(result.field.raw, np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1]))
    assert_frame_equal(result.distributions, pd.DataFrame([[0.4, 0.2], [0.6, 0.8]]))

    # original is not modified
    assert_equal(sd.field.raw, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))


def test_get_probability_field():

    sd = SpatialDistribution(
        field=core.VoxelData(np.array([0, 0, -1, 1, 1]), (1,)),
        distributions=pd.DataFrame([[0.4, 0.2],
                                    [0.6, 0.8]]),
        traits=pd.DataFrame({'size': ['Large', 'Small']}))

    assert_equal(sd.get_probability_field('size', 'Large').raw, np.array([0.4, 0.4, -1, 0.2, 0.2]))
    assert_equal(sd.get_probability_field('size', 'Small').raw, np.array([0.6, 0.6, -1, 0.8, 0.8]))


def test_from_probability_field():

    sd = SpatialDistribution.from_probability_field(
        core.VoxelData(np.array([0.4, 0.4, -1, 0.2, 0.2]), (1,)),
        'size', 'Large', 'Small'
    )

    # same as previous test but the columns in distributions are swapped
    # the reason for this is internal to numpy.unique but it doesn't really affect the result
    assert_equal(sd.field.raw, np.array([1, 1, -1, 0, 0]))
    assert_frame_equal(sd.distributions, pd.DataFrame([[0.2, 0.4], [0.8, 0.6]]))
    assert_frame_equal(sd.traits, pd.DataFrame({'size': ['Large', 'Small']}))

    # round trip
    assert_equal(sd.get_probability_field('size', 'Large').raw, np.array([0.4, 0.4, -1, 0.2, 0.2]))
    assert_equal(sd.get_probability_field('size', 'Small').raw, np.array([0.6, 0.6, -1, 0.8, 0.8]))
