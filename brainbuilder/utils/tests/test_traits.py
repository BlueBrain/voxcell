from nose.tools import eq_
from brainbuilder.utils import traits as tt
from brainbuilder.utils import genbrain as gb
from brainbuilder.utils.traits import SpatialDistribution as SD
import pandas as pd
import numpy as np
from numpy.testing import assert_equal
from pandas.util.testing import assert_frame_equal


def test_normalize_distribution_collection_empty():
    assert_frame_equal(tt.normalize_distribution_collection(pd.DataFrame()), pd.DataFrame())


def test_normalize_distribution_collection():
    r = tt.normalize_distribution_collection(pd.DataFrame({0: [0.1, 0.4],
                                                           1: [10, 10]}))

    assert_frame_equal(r, pd.DataFrame({0: [0.2, 0.8],
                                        1: [.5, .5]}))


NOFIELD = np.array([])


def test_split_distribution_collection_empty_0():
    probabilities = pd.DataFrame()
    attributes = ('name',)
    traits = pd.DataFrame(columns=attributes)

    eq_(SD(NOFIELD, probabilities, traits).split(attributes),
        {})


def test_split_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, distributions)
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_empty_2():
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame(index=[0]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame(index=[1]))
    # {('a',): SD(NOFIELD, [], traits), ('b',): SD(NOFIELD, [], traits)}


def test_split_distribution_collection_single_0():
    distributions = pd.DataFrame([{0: 1}])
    traits = pd.DataFrame([{'name': 'a'}])
    attributes = ('name',)

    # {('a',): SD(NOFIELD, [{0: 1}], traits)})

    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_single_1():
    distributions = pd.DataFrame([0.25, 0.75])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    attributes = ('name',)

    # {('a',): SD(NOFIELD, [{0: 1.0}], traits), ('b',): SD(NOFIELD, [{1: 1.0}], traits)})
    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame([1.], index=[1]))


def test_split_distribution_collection_single_2():
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'b'}])
    attributes = ('name',)

    res = SD(NOFIELD, distributions, traits).split(attributes)
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

    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.0]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([0.5, 0.5], index=[1, 2]))


def test_split_distribution_collection_multiattr_1():
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'},
                           {'name': 'b', 'type': 'y'}])

    attributes = ('name', 'type')

    res = SD(NOFIELD, distributions, traits).split(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y'), ('b', 'x')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res[('b', 'x')].distributions, pd.DataFrame([1.], index=[1]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([1.], index=[2]))

    # {('a', 'x'): SD(NOFIELD, [{0: 1.0}], traits),
    #  ('b', 'x'): SD(NOFIELD, [{1: 1.0}], traits),
    #  ('b', 'y'): SD(NOFIELD, [{2: 1.0}], traits)})


def test_reduce_distribution_collection_empty_0():
    sd = SD(NOFIELD, pd.DataFrame(), pd.DataFrame(columns=['name']))
    r = sd.reduce('name')
    assert_frame_equal(r.distributions, pd.DataFrame())


def test_reduce_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'}])
    dists = pd.DataFrame(index=traits.index)
    r = SD(NOFIELD, dists, traits).reduce('type')
    print r.distributions
    print r.traits
    assert_frame_equal(r.distributions, dists)


def test_reduce_distribution_collection_0():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'}])

    dists = pd.DataFrame([0.25, 0.75])

    r = SD(NOFIELD, dists, traits).reduce('type')
    assert_frame_equal(r.distributions,
                       pd.DataFrame([1.]))


def test_reduce_distribution_collection_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                          {'name': 'c', 'type': 'y'}])

    dists = pd.DataFrame([0.75, 0.25])

    r = SD(NOFIELD, dists, traits).reduce('type')
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

    r = SD(NOFIELD, dists, traits).reduce('type')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'type': ['y', 'x']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.7, 0.3]))

    r = SD(NOFIELD, dists, traits).reduce('name')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'name': ['a', 'c', 'b', 'd']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.1, 0.3, 0.2, 0.4]))


def check_assign_conditional(preassigned, expected):
    sdist = SD(field=gb.VoxelData(np.array([[[0]]]), (25, 25, 25)),
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
