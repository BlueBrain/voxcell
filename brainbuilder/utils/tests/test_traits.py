from nose.tools import eq_
from brainbuilder.utils import traits as tt
from brainbuilder.utils.traits import SpatialDistribution as SD


def test_normalize_distribution_empty():
    eq_(tt.normalize_probability_distribution({}), {})


def test_normalize_distribution_nowork():
    eq_(tt.normalize_probability_distribution({0: 0.5, 1: 0.5}), {0: 0.5, 1: 0.5})


def test_normalize_distribution():
    eq_(tt.normalize_probability_distribution({0: 0.1, 1: 0.4}), {0: 0.2, 1: 0.8})


def test_normalize_distribution_collection_empty_0():
    eq_(tt.normalize_distribution_collection([]), [])


def test_normalize_distribution_collection_empty_1():
    eq_(tt.normalize_distribution_collection([{}]), [{}])


def test_normalize_distribution_collection():
    eq_(tt.normalize_distribution_collection([{0: 0.1, 1: 0.4}, {0: 10, 1: 10}]),
                                              [{0: 0.2, 1: 0.8}, {0: 0.5, 1: 0.5}])


def test_split_distribution_collection_empty_0():
    probabilities = []
    traits_collection = []
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits_collection), attributes),
        {})


def test_split_distribution_collection_empty_1():
    probabilities = []
    traits = [{'name': 'a'}]
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a',): SD(None, [], traits)})


def test_split_distribution_collection_empty_2():
    probabilities = []
    traits = [{'name': 'a'}, {'name': 'b'}]
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a',): SD(None, [], traits), ('b',): SD(None, [], traits)})


def test_split_distribution_collection_single_0():
    probabilities = [{0: 1}]
    traits = [{'name': 'a'}]
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a',): SD(None, [{0: 1}], traits)})


def test_split_distribution_collection_single_1():
    probabilities = [{0: 0.25, 1: 0.75}]
    traits = [{'name': 'a'}, {'name': 'b'}]
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a',): SD(None, [{0: 1.0}], traits), ('b',): SD(None, [{1: 1.0}], traits)})


def test_split_distribution_collection_single_2():
    probabilities = [{0: 0.2, 1: 0.4, 2: 0.4}]
    traits = [{'name': 'a'}, {'name': 'b'}, {'name': 'b'}]
    attributes = ('name',)

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a',): SD(None, [{0: 1.0}], traits), ('b',): SD(None, [{1: 0.5, 2: 0.5}], traits)})


def test_split_distribution_collection_multiattr_0():
    probabilities = [{0: 0.2, 1: 0.4, 2: 0.4}]
    traits = [{'name': 'a', 'type': 'x'},
              {'name': 'b', 'type': 'y'},
              {'name': 'b', 'type': 'y'}]

    attributes = ('name', 'type')

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a', 'x'): SD(None, [{0: 1.0}], traits),
         ('b', 'y'): SD(None, [{1: 0.5, 2: 0.5}], traits)})


def test_split_distribution_collection_multiattr_1():
    probabilities = [{0: 0.2, 1: 0.4, 2: 0.4}]
    traits = [{'name': 'a', 'type': 'x'},
              {'name': 'b', 'type': 'x'},
              {'name': 'b', 'type': 'y'}]

    attributes = ('name', 'type')

    eq_(tt.split_distribution_collection(SD(None, probabilities, traits), attributes),
        {('a', 'x'): SD(None, [{0: 1.0}], traits),
         ('b', 'x'): SD(None, [{1: 1.0}], traits),
         ('b', 'y'): SD(None, [{2: 1.0}], traits)})


def test_reduce_distribution_collection_empty_0():
    eq_(tt.reduce_distribution_collection(SD(None, [], []), 'something'),
        SD(None, [], []))


def test_reduce_distribution_collection_empty_1():
    eq_(tt.reduce_distribution_collection(SD(None, [], [{'name': 'a', 'type': 'x'}]), 'type'),
        SD(None, [], [{'type': 'x'}]))


def test_reduce_distribution_collection_0():
    traits = [{'name': 'a', 'type': 'x'},
              {'name': 'b', 'type': 'x'}]

    dists = [{0: 0.25, 1: 0.75}]

    eq_(tt.reduce_distribution_collection(SD(None, dists, traits), 'type'),
        SD(None,
           [{0: 1.0}],
           [{'type': 'x'}]))


def test_reduce_distribution_collection_1():
    traits = [{'name': 'a', 'type': 'x'},
              {'name': 'c', 'type': 'y'}]

    dists = [{0: 0.75, 1: 0.25}]

    eq_(tt.reduce_distribution_collection(SD(None, dists, traits), 'type'),
        SD(None,
           [{0: 0.75, 1: 0.25}],
           [{'type': 'x'}, {'type': 'y'}]))


def test_reduce_distribution_collection_2():
    traits = [{'name': 'a', 'type': 'x'},
              {'name': 'b', 'type': 'x'},
              {'name': 'c', 'type': 'y'},
              {'name': 'd', 'type': 'y'}]

    dists = [{0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}]

    eq_(tt.reduce_distribution_collection(SD(None, dists, traits), 'type'),
        SD(None,
           [{0: 0.5, 1: 0.5}],
           [{'type': 'x'}, {'type': 'y'}]))
