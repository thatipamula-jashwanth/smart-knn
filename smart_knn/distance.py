import logging
import numpy as np
from numba import njit, prange


logger = logging.getLogger("smartknn.distance")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | smartknn.distance | %(message)s"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


def _ensure_f32_clean(X):
    if hasattr(X, "values"):
        X = X.values
    return np.nan_to_num(
        np.asarray(X, dtype=np.float32, order="C"),
        nan=0.0,
        posinf=1e9,
        neginf=-1e9
    )


def _validate_weights(weights, n_features, eps):
    w = np.asarray(weights, dtype=np.float32)

    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    if w.shape[0] != n_features:
        raise ValueError(
            f"Weight length {w.shape[0]} does not match feature dimension {n_features}"
        )
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, eps).astype(np.float32)

    s = float(np.sum(w))
    if s <= eps * len(w):
        logger.warning(
            "Effective weight sum is very small (sum=%.3e). Distance may be unstable.",
            s
        )
    return w


@njit(fastmath=True)
def _weighted_l2_single(a, b, w):
    s = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        s += diff * diff * w[i]
    return np.sqrt(s)


@njit(parallel=True, fastmath=True)
def _weighted_l2_batch(X, q, w):
    n, d = X.shape
    out = np.empty(n, dtype=np.float32)

    for i in prange(n):
        s = 0.0
        for j in range(d):
            diff = X[i, j] - q[j]
            s += diff * diff * w[j]
        out[i] = np.sqrt(s)

    return out


@njit(parallel=True, fastmath=True)
def _weighted_l2_multiquery(X, Q, w):
    nq, d = Q.shape
    nx = X.shape[0]

    out = np.empty((nq, nx), dtype=np.float32)

    block = 4096
    n_blocks = (nx + block - 1) // block

    for bi in prange(n_blocks):
        start = bi * block
        end = min(start + block, nx)

        for qi in range(nq):
            q = Q[qi]

            for xi in range(start, end):
                s = 0.0
                row = X[xi]

                for k in range(d):
                    diff = row[k] - q[k]
                    s += diff * diff * w[k]

                out[qi, xi] = np.sqrt(s)

    return out



def weighted_euclidean(a, b, weights, eps=1e-8):
    a = _ensure_f32_clean(a).reshape(-1)
    b = _ensure_f32_clean(b).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same shape: {a.shape} vs {b.shape}")

    w = _validate_weights(weights, a.shape[0], eps)
    return float(_weighted_l2_single(a, b, w))


def weighted_euclidean_batch(X, query, weights, eps=1e-8):
    X = _ensure_f32_clean(X)
    q = _ensure_f32_clean(query).reshape(-1)

    if X.shape[1] != q.shape[0]:
        raise ValueError(
            f"Query dimension {q.shape[0]} does not match X features {X.shape[1]}"
        )

    w = _validate_weights(weights, X.shape[1], eps)
    return _weighted_l2_batch(X, q, w)


def weighted_euclidean_multiquery(
    X, Q, weights, eps=1e-8, max_mem_bytes=1_000_000_000
):
    X = _ensure_f32_clean(X)
    Q = _ensure_f32_clean(Q)

    nx, d = X.shape
    nq = Q.shape[0]

    if Q.shape[1] != d:
        raise ValueError(
            f"Query features {Q.shape[1]} do not match X features {d}"
        )

    projected_bytes = int(nq) * int(nx) * 4
    if projected_bytes > max_mem_bytes:
        raise MemoryError(
            f"Projected memory too large: {projected_bytes} bytes "
            f"(limit={max_mem_bytes}). Use batching."
        )

    w = _validate_weights(weights, d, eps)
    return _weighted_l2_multiquery(X, Q, w)
