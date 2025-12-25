import numpy as np
import pytest

from smart_knn.data_processing import filter_low_weights


def test_filter_low_weights_basic():
    X = np.random.rand(10, 5).astype(np.float32)
    w = np.array([0.1, 0.5, 0.0, 0.2, 0.7], dtype=np.float32)

    X_f, w_f, mask = filter_low_weights(X, w, threshold=0.2, return_mask=True)

    assert mask.sum() == 3
    assert X_f.shape[1] == 3
    assert np.all(w_f >= 0.2)
    assert w_f.dtype == np.float32


def test_filter_low_weights_all_zero_with_min_features():
    X = np.random.rand(8, 4).astype(np.float32)
    w = np.zeros(4, dtype=np.float32)

    X_f, w_f, mask = filter_low_weights(
        X, w, threshold=0.5, min_features=2, return_mask=True
    )

    assert mask.sum() == 2
    assert X_f.shape[1] == 2
    assert w_f.shape == (2,)
    assert w_f.dtype == np.float32


def test_filter_low_weights_threshold_fallback_topk():
    X = np.random.rand(6, 4).astype(np.float32)
    w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    X_f, w_f, mask = filter_low_weights(
        X, w, threshold=0.5, min_features=2, return_mask=True
    )

    assert mask.sum() == 2
    assert np.allclose(np.sort(w_f), np.array([0.3, 0.4], dtype=np.float32))
    assert X_f.shape[1] == 2


def test_filter_low_weights_return_mask_shape():
    X = np.random.rand(5, 5).astype(np.float32)
    w = np.array([0.5, 0.0, 0.3, 0.8, 0.2], dtype=np.float32)

    X_f, w_f, mask = filter_low_weights(X, w, threshold=0.25, return_mask=True)

    assert mask.shape == (5,)
    assert mask.dtype == bool
    assert X_f.shape[1] == mask.sum()
    assert w_f.shape[0] == mask.sum()


def test_filter_low_weights_nan_inf_cleaning():
    X = np.array(
        [
            [1.0, np.nan, np.inf, -np.inf],
            [2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    w = np.array([0.1, np.nan, np.inf, -np.inf], dtype=np.float32)

    X_f, w_f, mask = filter_low_weights(
        X, w, threshold=0.0, return_mask=True
    )

    assert np.isfinite(X_f).all()
    assert np.isfinite(w_f).all()
    assert np.all(w_f >= 0)
    assert X_f.dtype == np.float32


def test_filter_low_weights_dim_mismatch():
    X = np.random.rand(10, 4)
    w = np.random.rand(3)

    with pytest.raises(ValueError):
        filter_low_weights(X, w)


def test_filter_low_weights_invalid_X_ndim():
    X = np.random.rand(5)  # not 2D
    w = np.random.rand(5)

    with pytest.raises(ValueError):
        filter_low_weights(X, w)


def test_filter_low_weights_all_filtered_runtime_error():
    """
    CONTRACT:
    - If threshold removes everything AND min_features == 0,
      this is a user error and should raise.
    """
    X = np.random.rand(10, 3).astype(np.float32)
    w = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    with pytest.raises(RuntimeError):
        filter_low_weights(X, w, threshold=1.0, min_features=0)
