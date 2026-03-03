import numpy as np
import warnings
from sklearn.metrics import r2_score

try:
    from .backends.perf_backend import AnnBackend
    ANN_AVAILABLE = True
except Exception:
    AnnBackend = None
    ANN_AVAILABLE = False


def _validate_ann_regression(self, max_samples=1024):

    if not ANN_AVAILABLE:
        raise RuntimeError("ANN backend not available.")

    X_full, y_full = self.X_, self.y_
    n = X_full.shape[0]

    if n > max_samples:
        sel = np.random.choice(n, max_samples, replace=False)
        X = X_full[sel]
        y = y_full[sel]
    else:
        X, y = X_full, y_full

    ann = AnnBackend(
        X,
        use_ivf=False,
        use_gpu=False,
        silent=True
    )

    idx_mat, dist_mat = ann.kneighbors_batch(X, self.k)

    w = 1.0 / np.maximum(dist_mat, 1e-9)
    y_neighbors = y[idx_mat]
    preds = np.sum(y_neighbors * w, axis=1) / np.sum(w, axis=1)

    return r2_score(y, preds)


def _kneighbors_batch(self, Xq):

    Xq = np.asarray(Xq, dtype=np.float32)
    if Xq.ndim == 1:
        Xq = Xq.reshape(1, -1)

    if not np.isfinite(Xq).all():
        warnings.warn(
            "NaN/Inf Detected in Query — APPLYING SAFE NORMALIZATION.",
            RuntimeWarning,
        )

    Xq = np.nan_to_num(
        Xq, nan=self.mean_, posinf=self.mean_, neginf=self.mean_
    )

    Xq = (Xq - self.mean_) / np.maximum(self.std_, 1e-12)
    Q = Xq[:, self.feature_mask_]

    approx_idx, _ = self.backend.kneighbors_batch(Q, self.k * 5)

    Xc = self.X_[approx_idx]
    diff = Xc - Q[:, None, :]
    dist = np.sqrt(np.sum((diff * diff) * self.weights_, axis=2))

    top = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]

    return (
        np.take_along_axis(approx_idx, top, axis=1),
        np.take_along_axis(dist, top, axis=1)
    )