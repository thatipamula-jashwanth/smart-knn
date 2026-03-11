import numpy as np
from sklearn.metrics import r2_score

try:
    from .backends.perf_backend import AnnBackend
    ANN_AVAILABLE = True
except ImportError:
    AnnBackend = None
    ANN_AVAILABLE = False


def _validate_ann_regression(self, max_samples=1024):

    if not ANN_AVAILABLE:
        raise RuntimeError("ANN backend not available.")

    X_full = self.X_
    y_full = self.y_

    n = X_full.shape[0]

    if n > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, max_samples, replace=False)
        X = X_full[sel]
        y = y_full[sel]
    else:
        X = X_full
        y = y_full

    ann = AnnBackend(
        X,
        use_ivf=False,
        use_gpu=False,
        silent=True
    )

    idx_mat, dist_mat = ann.kneighbors_batch(X, self.k)

    if idx_mat.shape[1] < self.k:
        raise RuntimeError("ANN backend returned insufficient neighbors")

    if not np.isfinite(dist_mat).all():
        raise RuntimeError("ANN backend returned invalid distances")

    w = 1.0 / np.maximum(dist_mat, 1e-9)

    y_neighbors = y[idx_mat]

    preds = np.sum(y_neighbors * w, axis=1) / np.sum(w, axis=1)

    return r2_score(y, preds)


def _kneighbors_batch(self, Xq):

    Q = np.asarray(Xq, dtype=np.float32)

    if Q.ndim == 1:
        Q = Q.reshape(1, -1)

    n_samples = self.X_.shape[0]

    cand_k = min(self.k * 5, n_samples)

    approx_idx, _ = self.backend.kneighbors_batch(Q, cand_k)

    Xc = self.X_[approx_idx]

    diff = Xc - Q[:, None, :]

    dist = np.sqrt(np.sum(diff * diff * self.weights_, axis=2))

    if getattr(self, "global_u_", None) is not None:

        proj_q = Q @ self.global_u_
        proj_x = self.X_global_[approx_idx]

        diff_global = proj_x - proj_q[:, None, :]

        global_term = np.sqrt(np.sum(diff_global * diff_global, axis=2))

        dist = dist + self.global_lambda * global_term

    top = np.argpartition(dist, self.k - 1, axis=1)[:, :self.k]

    row_ids = np.arange(top.shape[0])[:, None]
    top = top[row_ids, np.argsort(dist[row_ids, top])]

    final_idx = np.take_along_axis(approx_idx, top, axis=1)
    final_dist = np.take_along_axis(dist, top, axis=1)

    return final_idx, final_dist