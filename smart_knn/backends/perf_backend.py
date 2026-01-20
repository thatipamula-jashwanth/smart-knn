import numpy as np
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

logger = logging.getLogger("Smartknn.Annbackend")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | SmartKNN.ANN | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class AnnBackend:
    def __init__(self, X_weighted, nprobe=None, nlist=None, use_gpu=False, use_ivf=True, silent=False):

        self.silent = silent
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not installed. Install with:\n  pip install faiss-cpu\nor: pip install faiss-gpu"
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
                    self.nlist = min(2048, max(64, n // 40))
                else:
                    if n <= 5_000_000:
                        self.nlist = 512
                    else:
                        self.nlist = 256
            else:
                self.nlist = int(nlist)
        else:
            self.nlist = 1


        if nprobe is None:
            self.nprobe = min(8, max(1, self.nlist // 10))
        else:
            self.nprobe = int(nprobe)

        if self.use_ivf:
            self.nprobe = min(max(1, self.nprobe), self.nlist)

        quantizer = faiss.IndexFlatL2(d)
        if self.use_ivf:
            index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
            index.train(X)
            if not index.is_trained:
                raise RuntimeError("FAISS IVF training failed")
            index.add(X)
            index.nprobe = self.nprobe
            if not silent:
                logger.info(f"IVF index ready | nlist={self.nlist} | nprobe={index.nprobe}")
                logger.warning("ANN recall depends on nlist/nprobe — tune for your dataset")
        else:
            index = faiss.IndexFlatL2(d)
            index.add(X)
            if not silent:
                logger.info("FlatL2 index ready (exact search)")

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                if not silent:
                    logger.info("FAISS GPU enabled")
            except Exception as e:
                if not silent:
                    logger.warning(f"GPU unavailable — CPU fallback ({e})")
                self.use_gpu = False

        self.index = index
        if silent:
            logger.debug("ANN recall validated.")
        else:
            logger.info(
                f"ANN backend ready | samples={n} | features={d} | ivf={self.use_ivf} | gpu={self.use_gpu}"
            )

    def set_nprobe(self, nprobe):
        if not self.use_ivf:
            if not self.silent:
                logger.warning("nprobe has no effect for FlatL2 index")
            return
        nprobe = max(1, min(int(nprobe), self.nlist))
        self.index.nprobe = nprobe
        self.nprobe = nprobe
        if not self.silent:
            logger.info(f"nprobe updated → {self.nprobe}")

    def search(self, query, k):
        if k <= 0:
            raise ValueError("k must be >= 1")
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(f"Query dimension mismatch: got {q.shape}, expected ({self.dim},)")
        if not np.isfinite(q).all():
            raise ValueError("Query contains NaN/Inf")
        q = q.reshape(1, -1)
        k = min(k, self.X.shape[0])
        dist, idx = self.index.search(q, k)
        idx = np.where(idx[0] == -1, 0, idx[0])
        return idx, dist[0]

    def kneighbors(self, query, k):
        return self.search(query, k)

    def kneighbors_batch(self, Q, k):
        if k <= 0:
            raise ValueError("k must be >= 1")
        Q = np.asarray(Q, dtype=np.float32)
        if Q.ndim != 2 or Q.shape[1] != self.dim:
            raise ValueError(f"Query matrix shape invalid: got {Q.shape}, expected (*, {self.dim})")
        if not np.isfinite(Q).all():
            raise ValueError("Query matrix contains NaN/Inf")
        if self.use_gpu and Q.shape[0] > 10_000:
            raise ValueError("Batch too large for GPU ANN backend")
        k = min(k, self.X.shape[0])
        dist, idx = self.index.search(Q, k)
        idx = np.where(idx == -1, 0, idx)
        return idx, dist
