import numpy as np
import pytest

from smart_knn import SmartKNN


def test_classification_basic_binary():
    np.random.seed(42)

    X = np.random.rand(200, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    model = SmartKNN(k=5)
    model.fit(X, y)

    preds = model.predict(X)

    assert model.is_classification_ is True
    assert model.classes_.tolist() == [0, 1]
    assert preds.shape == y.shape
    assert preds.dtype == y.dtype

    acc = np.mean(preds == y)
    assert acc > 0.85, f"Low classification accuracy: {acc}"


def test_classification_multiclass():
    np.random.seed(0)

    X = np.random.rand(300, 3).astype(np.float32)

    y = np.zeros(300, dtype=int)
    y[X[:, 0] > 0.66] = 2
    y[(X[:, 0] > 0.33) & (X[:, 0] <= 0.66)] = 1

    model = SmartKNN(k=7)
    model.fit(X, y)

    preds = model.predict(X)

    assert model.is_classification_ is True
    assert set(model.classes_) == {0, 1, 2}
    assert preds.shape == y.shape

    acc = np.mean(preds == y)
    assert acc > 0.75


def test_force_classification_overrides_regression():
    np.random.seed(42)

    X = np.random.rand(100, 2).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = SmartKNN(force_classification=True)
    model.fit(X, y)

    assert model.is_classification_ is True
    assert model.classes_ is not None

    preds = model.predict(X)
    assert preds.ndim == 1


def test_classification_nan_inf_query_raises():
    """
    SmartKNN contract: training data may contain NaN/Inf (sanitized),
    but queries with NaN/Inf must raise ValueError.
    """
    X = np.array(
        [
            [1.0, np.nan, 0.2],
            [0.9, np.inf, 0.1],
            [0.1, -np.inf, 0.9],
            [0.2, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1], dtype=int)

    model = SmartKNN(k=2)
    model.fit(X, y)

    q = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)

    with pytest.raises(ValueError):
        model.predict(q)


def test_classification_batch_vs_single():
    np.random.seed(1)

    X = np.random.rand(120, 5).astype(np.float32)
    y = (X[:, 2] > 0.5).astype(int)

    model = SmartKNN(k=3)
    model.fit(X, y)

    q = X[:10]
    preds_batch = model.predict(q)
    preds_single = np.array([model.predict(x)[0] for x in q])

    assert np.array_equal(preds_batch, preds_single)
