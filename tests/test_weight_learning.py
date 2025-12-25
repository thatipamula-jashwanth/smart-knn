import numpy as np
import pytest

from smart_knn.weight_learning import learn_feature_weights

def test_learn_feature_weights_basic_properties():
    np.random.seed(42)

    X = np.random.rand(200, 5).astype(np.float32)
    y = (3 * X[:, 2] + 0.1 * np.random.randn(200)).astype(np.float32)

    w = learn_feature_weights(X, y)

    assert isinstance(w, np.ndarray)
    assert w.shape == (X.shape[1],)
    assert w.dtype == np.float32

    assert np.isfinite(w).all()
    assert np.all(w > 0)
    assert np.isclose(np.sum(w), 1.0, atol=1e-5)


def test_dominant_feature_gets_higher_weight():
    np.random.seed(0)

    X = np.random.randn(300, 4).astype(np.float32)
    y = 5 * X[:, 1] + 0.05 * np.random.randn(300)

    w = learn_feature_weights(X, y)

    assert np.argmax(w) == 1
    assert w[1] > np.mean(np.delete(w, 1))


def test_constant_feature_not_dominant():
    np.random.seed(1)

    X = np.random.rand(200, 4).astype(np.float32)
    X[:, 0] = 1.0  
    X[:, 3] = 5.0  

    y = 2 * X[:, 1] + 0.01 * np.random.randn(200)

    w = learn_feature_weights(X, y)

    assert w[1] == np.max(w)
    assert w[0] < w[1]
    assert w[3] < w[1]


def test_nan_inf_safe_inputs():
    X = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, np.inf, 4.0],
            [3.0, -np.inf, 5.0],
            [4.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    w = learn_feature_weights(X, y)

    assert np.isfinite(w).all()
    assert np.isclose(np.sum(w), 1.0, atol=1e-5)


def test_single_feature_input():
    np.random.seed(3)

    X = np.random.rand(100, 1).astype(np.float32)
    y = 4 * X[:, 0] + 0.01 * np.random.randn(100)

    w = learn_feature_weights(X, y)

    assert w.shape == (1,)
    assert np.isclose(w[0], 1.0, atol=1e-6)


def test_large_dataset_subsampling_path():
    np.random.seed(4)

    X = np.random.rand(60000, 3).astype(np.float32)
    y = X[:, 0] + 0.1 * np.random.randn(60000)

    w = learn_feature_weights(X, y)

    assert w.shape == (3,)
    assert np.isfinite(w).all()
    assert np.isclose(np.sum(w), 1.0, atol=1e-5)
