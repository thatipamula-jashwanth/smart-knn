import numpy as np
import pytest

from smart_knn.distance import (
    weighted_euclidean,
    weighted_euclidean_batch,
    weighted_euclidean_multiquery,
)



def test_weighted_euclidean_basic():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    w = np.array([1.0, 1.0, 1.0])

    d = weighted_euclidean(a, b, w)

    expected = np.sqrt((1**2 + 2**2 + 3**2))
    assert isinstance(d, float)
    assert np.isfinite(d)
    assert np.isclose(d, expected, atol=1e-6)


def test_weighted_euclidean_nan_inf_safe():
    a = np.array([np.nan, np.inf, -np.inf])
    b = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])

    d = weighted_euclidean(a, b, w)

    assert isinstance(d, float)
    assert np.isfinite(d)


def test_weighted_euclidean_shape_mismatch():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        weighted_euclidean(a, b, w)


def test_weighted_euclidean_invalid_weights():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 3.0])

    with pytest.raises(ValueError):
        weighted_euclidean(a, b, weights=[1.0, -1.0])



def test_weighted_euclidean_batch_basic():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    q = np.array([1.0, 1.0])
    w = np.array([1.0, 1.0])

    d = weighted_euclidean_batch(X, q, w)

    assert d.shape == (3,)
    assert d.dtype == np.float32
    assert np.isfinite(d).all()

    assert np.isclose(d[1], 0.0, atol=1e-6)


def test_weighted_euclidean_batch_dim_mismatch():
    X = np.random.rand(5, 3)
    q = np.random.rand(2)
    w = np.ones(3)

    with pytest.raises(ValueError):
        weighted_euclidean_batch(X, q, w)



def test_weighted_euclidean_multiquery_basic():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    Q = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    w = np.array([1.0, 1.0])

    D = weighted_euclidean_multiquery(X, Q, w)

    assert D.shape == (2, 2)
    assert D.dtype == np.float32
    assert np.isfinite(D).all()

    assert np.isclose(D[0, 0], 0.0)
    assert np.isclose(D[1, 1], 0.0)


def test_weighted_euclidean_multiquery_dim_mismatch():
    X = np.random.rand(10, 4)
    Q = np.random.rand(3, 5)
    w = np.ones(4)

    with pytest.raises(ValueError):
        weighted_euclidean_multiquery(X, Q, w)


def test_weighted_euclidean_multiquery_memory_guard():
    X = np.random.rand(1000, 10)
    Q = np.random.rand(1000, 10)
    w = np.ones(10)

    with pytest.raises(MemoryError):
        weighted_euclidean_multiquery(
            X, Q, w, max_mem_bytes=1024 
        )
