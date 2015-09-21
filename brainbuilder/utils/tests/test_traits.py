from nose.tools import eq_
from brainbuilder.utils import traits as tt
from brainbuilder.utils.traits import SpatialDistribution as SD
import pandas as pd
from pandas.util.testing import assert_frame_equal


def test_normalize_distribution_collection_empty():
    assert_frame_equal(tt.normalize_distribution_collection(pd.DataFrame()), pd.DataFrame())


def test_normalize_distribution_collection():
    r = tt.normalize_distribution_collection(pd.DataFrame({0: [0.1, 0.4],
                                                           1: [10, 10]}))

    assert_frame_equal(r, pd.DataFrame({0: [0.2, 0.8],
                                        1: [.5, .5]}))


def test_split_distribution_collection_empty_0():
    probabilities = pd.DataFrame()
    attributes = ('name',)
    traits = pd.DataFrame(columns=attributes)

    eq_(SD(None, probabilities, traits, None).split_distribution_collection(attributes),
        {})


def test_split_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    field = None
    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, distributions)
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_empty_2():
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    distributions = pd.DataFrame(index=traits.index)
    attributes = ('name',)

    field = None
    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame())
    assert_frame_equal(res['b'].distributions, pd.DataFrame())
    # {('a',): SD(None, [], traits, None), ('b',): SD(None, [], traits, None)}


def test_split_distribution_collection_single_0():
    field = None
    distributions = pd.DataFrame([{0: 1}])
    traits = pd.DataFrame([{'name': 'a'}])
    attributes = ('name',)

    # {('a',): SD(None, [{0: 1}], traits, None)})

    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), ['a'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['a'].traits, traits)


def test_split_distribution_collection_single_1():
    field = None
    distributions = pd.DataFrame([0.25, 0.75])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}])
    attributes = ('name',)

    # {('a',): SD(None, [{0: 1.0}], traits, None), ('b',): SD(None, [{1: 1.0}], traits, None)})
    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame([1.], index=[1]))


def test_split_distribution_collection_single_2():
    field = None
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a'}, {'name': 'b'}, {'name': 'b'}])
    attributes = ('name',)

    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), ['a', 'b'])
    assert_frame_equal(res['a'].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res['b'].distributions, pd.DataFrame([0.5, 0.5], index=[1, 2]))


def test_split_distribution_collection_multiattr_0():
    field = None
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'y'},
                           {'name': 'b', 'type': 'y'}])

    attributes = ('name', 'type')

    # {('a', 'x'): SD(None, [{0: 1.0}], traits, None),
    #  ('b', 'y'): SD(None, [{1: 0.5, 2: 0.5}], traits, None)})

    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.0]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([0.5, 0.5], index=[1, 2]))


def test_split_distribution_collection_multiattr_1():
    field = None
    distributions = pd.DataFrame([0.2, 0.4, 0.4])
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'},
                           {'name': 'b', 'type': 'y'}])

    attributes = ('name', 'type')

    res = SD(field, distributions, traits, None).split_distribution_collection(attributes)
    eq_(res.keys(), [('a', 'x'), ('b', 'y'), ('b', 'x')])
    assert_frame_equal(res[('a', 'x')].distributions, pd.DataFrame([1.]))
    assert_frame_equal(res[('b', 'x')].distributions, pd.DataFrame([1.], index=[1]))
    assert_frame_equal(res[('b', 'y')].distributions, pd.DataFrame([1.], index=[2]))

    # {('a', 'x'): SD(None, [{0: 1.0}], traits, None),
    #  ('b', 'x'): SD(None, [{1: 1.0}], traits, None),
    #  ('b', 'y'): SD(None, [{2: 1.0}], traits, None)})


def test_reduce_distribution_collection_empty_0():
    sd = SD(None, pd.DataFrame(), pd.DataFrame(columns=['name']), None)
    r = sd.reduce_distribution_collection('name')
    assert_frame_equal(r.distributions, pd.DataFrame())


def test_reduce_distribution_collection_empty_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'}])
    dists = pd.DataFrame(index=traits.index)
    r = SD(None, dists, traits, None).reduce_distribution_collection('type')
    print r.distributions
    print r.traits
    assert_frame_equal(r.distributions, dists)


def test_reduce_distribution_collection_0():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                           {'name': 'b', 'type': 'x'}])

    dists = pd.DataFrame([0.25, 0.75])

    r = SD(None, dists, traits, None).reduce_distribution_collection('type')
    assert_frame_equal(r.distributions,
                       pd.DataFrame([1.]))


def test_reduce_distribution_collection_1():
    traits = pd.DataFrame([{'name': 'a', 'type': 'x'},
                          {'name': 'c', 'type': 'y'}])

    dists = pd.DataFrame([0.75, 0.25])

    r = SD(None, dists, traits, None).reduce_distribution_collection('type')
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

    r = SD(None, dists, traits, None).reduce_distribution_collection('type')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'type': ['y', 'x']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.7, 0.3]))

    r = SD(None, dists, traits, None).reduce_distribution_collection('name')
    assert_frame_equal(r.traits,
                       pd.DataFrame({'name': ['a', 'c', 'b', 'd']}))
    assert_frame_equal(r.distributions,
                       pd.DataFrame([0.1, 0.3, 0.2, 0.4]))
