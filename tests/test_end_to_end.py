import numpy as np
import pytest
import warnings

from smart_knn import SmartKNN


def test_end_to_end_regression_basic():
    np.random.seed(42)

    X = np.random.rand(200, 4).astype(np.float32)
    y = (3 * X[:, 2] + 2 * X[:, 1] + 0.1 * np.random.randn(200)).astype(np.float32)

    model = SmartKNN(k=5)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert preds.dtype in (np.float32, np.float64)

    mse = np.mean((preds - y) ** 2)
    assert np.isfinite(mse)
    assert mse < 0.2, f"MSE too high: {mse}"


def test_end_to_end_nan_inf_query_warns():
    """
    SmartKNN v2 CONTRACT:
    - Training data may contain NaN/Inf (sanitized)
    - Query data with NaN/Inf triggers a warning and produces finite predictions.
    """
    X = np.array([
        [1.0, np.nan, 5.0],
        [2.0, np.inf, 6.0],
        [3.0, -np.inf, 7.0],
    ], dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    model = SmartKNN(k=2)
    model.fit(X, y)

    q = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preds = model.predict(q)

        # Ensure a warning was issued for NaN/Inf
        assert any("NaN/Inf detected" in str(wi.message) for wi in w), \
            "Expected a warning about NaN/Inf in query"

        # Predictions are finite
        assert np.isfinite(preds).all()


def test_feature_filtering_threshold():
    np.random.seed(42)

    X = np.random.rand(100, 6).astype(np.float32)
    y = (5 * X[:, 0] + 0.05 * np.random.randn(100)).astype(np.float32)

    model = SmartKNN(k=3, weight_threshold=0.2)
    model.fit(X, y)

    assert hasattr(model, "feature_mask_")
    assert model.feature_mask_.dtype == bool
    assert model.feature_mask_.sum() >= 1
    assert model.X_.shape[1] == model.feature_mask_.sum()


def test_predict_single_query_shape():
    np.random.seed(42)

    X = np.random.rand(50, 4).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)

    model = SmartKNN()
    model.fit(X, y)

    q = np.random.rand(4).astype(np.float32)
    pred = model.predict(q)

    assert pred.shape == (1,)
    assert np.isfinite(pred).all()


def test_kneighbors_returns_sorted_distances():
    np.random.seed(42)

    X = np.random.rand(40, 3).astype(np.float32)
    y = np.random.randn(40).astype(np.float32)

    model = SmartKNN(k=5)
    model.fit(X, y)

    q = np.random.rand(3).astype(np.float32)
    idx, dists = model._kneighbors_batch(q)

    assert idx.shape[1] == model.k
    assert dists.shape[1] == model.k
    assert np.all(np.diff(np.sort(dists[0])) >= 0)


def test_predict_not_fitted():
    model = SmartKNN()
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])


def test_query_dim_mismatch():
    X = np.random.rand(20, 5).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)

    model = SmartKNN()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict([1, 2, 3])


def test_predict_batch_queries():
    np.random.seed(42)

    X = np.random.rand(100, 4).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = SmartKNN(k=3)
    model.fit(X, y)

    Q = np.random.rand(8, 4).astype(np.float32)
    preds = model.predict(Q)

    assert preds.shape == (8,)
    assert preds.dtype in (np.float32, np.float64)
    assert np.isfinite(preds).all()
