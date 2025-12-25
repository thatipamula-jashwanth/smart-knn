import numpy as np
import logging
from numba import njit, prange

logger = logging.getLogger("smartknn.brutebackend")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Brute] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)  
logger.propagate = False


@njit(parallel=True, fastmath=True)
def _weighted_euclidean_single(X, q, w):
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
def _weighted_euclidean_batch(X, Q, w):
    B, d = Q.shape
    n = X.shape[0]
    out = np.empty((B, n), dtype=np.float32)

    for b in prange(B):         
        q = Q[b]
        for i in range(n):       
            s = 0.0
            for j in range(d):
                diff = X[i, j] - q[j]
                s += diff * diff * w[j]
            out[b, i] = np.sqrt(s)

    return out


class BruteBackend:

    def __init__(self, X, weights, debug=False):
        self.X = np.asarray(X, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.debug = bool(debug)

        if self.X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.X.shape[1]:
            raise ValueError("Weights / feature mismatch")

        logger.warning(
            f"samples={self.X.shape[0]} | "
            f"features={self.X.shape[1]}"
        )


    def search(self, query, k):
        q = np.asarray(query, dtype=np.float32)

        if q.ndim != 1 or q.shape[0] != self.X.shape[1]:
            raise ValueError("Query feature mismatch")

        dists = _weighted_euclidean_single(self.X, q, self.weights)

        k = min(k, dists.shape[0])
        idx = np.argpartition(dists, k - 1)[:k]
        idx = idx[np.argsort(dists[idx])]

        return idx, dists[idx]


    def kneighbors(self, query, k):
        return self.search(query, k)


    def kneighbors_batch(self, Q, k):
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        if Q.shape[1] != self.X.shape[1]:
            raise ValueError("Query feature mismatch")

       
        dist_matrix = _weighted_euclidean_batch(self.X, Q, self.weights)

        k = min(k, self.X.shape[0])
        topk_idx = np.argpartition(dist_matrix, k - 1, axis=1)[:, :k]

        topk_dist = np.take_along_axis(dist_matrix, topk_idx, axis=1)
        order = np.argsort(topk_dist, axis=1)

        sorted_idx = np.take_along_axis(topk_idx, order, axis=1)
        sorted_dist = np.take_along_axis(topk_dist, order, axis=1)

        return sorted_idx, sorted_dist
