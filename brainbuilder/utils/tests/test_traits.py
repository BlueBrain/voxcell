from nose.tools import eq_
from brainbuilder.utils import traits as tt


def test_normalize_distribution_empty():
    eq_(tt.normalize_distribution({}), {})


def test_normalize_distribution_nowork():
    eq_(tt.normalize_distribution({0: 0.5, 1: 0.5}), {0: 0.5, 1: 0.5})


def test_normalize_distribution():
    eq_(tt.normalize_distribution({0: 0.1, 1: 0.4}), {0: 0.2, 1: 0.8})


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
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {})


def test_split_distribution_collection_empty_1():
    probabilities = []
    traits_collection = [{'name': 'a'}]
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {'a': []})


def test_split_distribution_collection_empty_2():
    probabilities = []
    traits_collection = [{'name': 'a'}, {'name': 'b'}]
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {'a': [], 'b': []})


def test_split_distribution_collection_single_0():
    probabilities = [{0: 1}]
    traits_collection = [{'name': 'a'}]
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {'a': [{0: 1}]})


def test_split_distribution_collection_single_1():
    probabilities = [{0: 0.25, 1: 0.75}]
    traits_collection = [{'name': 'a'}, {'name': 'b'}]
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {'a': [{0: 1.0}], 'b': [{1: 1.0}]})


def test_split_distribution_collection_single_2():
    probabilities = [{0: 0.2, 1: 0.4, 2: 0.4}]
    traits_collection = [{'name': 'a'}, {'name': 'b'}, {'name': 'b'}]
    attribute = 'name'

    eq_(tt.split_distribution_collection(probabilities, traits_collection, attribute),
        {'a': [{0: 1.0}], 'b': [{1: 0.5, 2: 0.5}]})
