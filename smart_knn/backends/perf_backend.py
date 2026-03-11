import numpy as np
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnnBackend:

    def __init__(self, X_weighted, nprobe=None, nlist=None, use_gpu=False, use_ivf=True, silent=False):

        self.silent = silent

        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not installed. Install with:\n"
                "  pip install faiss-cpu\n"
                "or: pip install faiss-gpu"
            )

        X = np.ascontiguousarray(X_weighted, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[0] == 0:
            raise ValueError("Empty dataset is not allowed")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf")

        n, d = X.shape

        self.X = X
        self.dim = d
        self.use_gpu = bool(use_gpu)
        self.use_ivf = bool(use_ivf)

        if self.use_ivf:
            if nlist is None:
                if n <= 1_000_000:
                    self.nlist = min(2048, max(64, n // 50))
                elif n <= 5_000_000:
                    self.nlist = 512
                else:
                    self.nlist = 256
            else:
                self.nlist = int(nlist)
        else:
            self.nlist = 1

        if nprobe is None:
            probe = int(np.sqrt(self.nlist))
        else:
            probe = int(nprobe)

        probe = max(1, probe)
        probe = min(probe, 8)  

        if self.use_ivf:
            probe = min(probe, self.nlist)

        self.nprobe = probe

        quantizer = faiss.IndexFlatL2(d)

        if self.use_ivf:

            index = faiss.IndexIVFFlat(quantizer, d, self.nlist)

            if not index.is_trained:

                train_size = min(200_000, n)

                if train_size < n:
                    rng = np.random.default_rng(42)
                    train_idx = rng.choice(n, train_size, replace=False)
                    train_data = X[train_idx]
                else:
                    train_data = X

                index.train(train_data)

            index.add(X)
            index.nprobe = self.nprobe

            if not silent:
                logger.info("IVF index built")

        else:

            index = faiss.IndexFlatL2(d)
            index.add(X)

            if not silent:
                logger.info("FlatL2 index ready (exact search)")

        if self.use_gpu:

            try:
                self.gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self.gpu_res, 0, index)

                if not silent:
                    logger.info("FAISS GPU enabled")

            except Exception as e:

                if not silent:
                    logger.warning(f"GPU unavailable — CPU fallback ({e})")

                self.use_gpu = False

        self.index = index

        if not silent:
            logger.info(f"Samples={n} | Features={d} | GPU={self.use_gpu}")

    def set_nprobe(self, nprobe):

        if not self.use_ivf:
            if not self.silent:
                logger.warning("nprobe has no effect for FlatL2 index")
            return

        nprobe = int(nprobe)

        nprobe = max(1, min(nprobe, self.nlist))
        nprobe = min(nprobe, 8)

        self.index.nprobe = nprobe
        self.nprobe = nprobe

        if not self.silent:
            logger.info(f"nprobe updated → {self.nprobe}")


    def _fix_invalid_indices(self, idx):

        if np.any(idx == -1):

            if not self.silent:
                logger.warning("FAISS returned incomplete neighbors — filling missing indices")

            first_valid = idx[:, 0:1]
            idx = np.where(idx == -1, first_valid, idx)

        return idx

    def search(self, query, k):

        if k <= 0:
            raise ValueError("k must be >= 1")

        q = np.ascontiguousarray(query, dtype=np.float32)

        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(
                f"Query dimension mismatch: got {q.shape}, expected ({self.dim},)"
            )

        if not np.isfinite(q).all():
            raise ValueError("Query contains NaN/Inf")

        q = q.reshape(1, -1)

        k = min(k, self.X.shape[0])

        dist, idx = self.index.search(q, k)

        idx = self._fix_invalid_indices(idx)

        return idx[0], dist[0]

    def kneighbors(self, query, k):
        return self.search(query, k)

    def kneighbors_batch(self, Q, k):

        if k <= 0:
            raise ValueError("k must be >= 1")

        Q = np.ascontiguousarray(Q, dtype=np.float32)

        if Q.ndim != 2 or Q.shape[1] != self.dim:
            raise ValueError(
                f"Query matrix shape invalid: got {Q.shape}, expected (*, {self.dim})"
            )

        if not np.isfinite(Q).all():
            raise ValueError("Query matrix contains NaN/Inf")

        if self.use_gpu and Q.shape[0] > 10_000 and not self.silent:
            logger.warning("Large GPU batch may consume significant memory")

        k = min(k, self.X.shape[0])

        dist, idx = self.index.search(Q, k)

        idx = self._fix_invalid_indices(idx)

        return idx, dist