import logging
import threading
import warnings
import numpy as np
from sklearn.metrics import r2_score

from .weight_learning import learn_feature_weights
from .utils import normalize, clip_weights
from .data_processing import filter_low_weights
from .backends.brute_backend import BruteBackend

try:
    from .backends.perf_backend import AnnBackend
    ANN_AVAILABLE = True
except Exception:
    ANN_AVAILABLE = False

logger = logging.getLogger("SmartKNN")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


class SmartKNN:
    __slots__ = (
        "k", "weight_threshold", "alpha", "beta", "gamma",
        "force_classification",
        "backend_mode", "use_gpu",
        "ann_quality_check", "ann_min_r2",
        "_lock", "fitted",
        "mean_", "std_",
        "feature_mask_", "X_", "weights_", "y_",
        "n_features_", "is_classification_", "classes_",
        "backend"
    )

    def __init__(
        self,
        k=5,
        weight_threshold=0.0,
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        force_classification=None,
        backend="auto",
        use_gpu=False,
        ann_quality_check=True,
        ann_min_r2=0.0
    ):
        if backend not in ("auto", "brute", "ann"):
            raise ValueError("backend must be 'auto', 'brute', or 'ann'")

        self.k = int(k)
        self.weight_threshold = float(weight_threshold)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

        self.force_classification = force_classification
        self.backend_mode = backend
        self.use_gpu = bool(use_gpu)

        self.ann_quality_check = bool(ann_quality_check)
        self.ann_min_r2 = float(ann_min_r2)

        self.fitted = False
        self._lock = threading.Lock()

    def __sklearn_is_fitted__(self):
        return getattr(self, "fitted", False)

    def get_params(self, deep=True):
        return {
            "k": self.k,
            "weight_threshold": self.weight_threshold,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "force_classification": self.force_classification,
            "backend": self.backend_mode,
            "use_gpu": self.use_gpu,
            "ann_quality_check": self.ann_quality_check,
            "ann_min_r2": self.ann_min_r2,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == "backend":
                setattr(self, "backend_mode", value)
            else:
                setattr(self, key, value)
        return self

    def _validate_schema_array(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if not np.isfinite(X).all():
            logger.warning("NaN/Inf detected in X — applying safe normalization.")
        if y is not None and not np.isfinite(y).all():
            raise ValueError("y contains NaN/Inf.")

    def _infer_classification(self, y):
        if self.force_classification is True:
            return True
        if self.force_classification is False:
            return False
        return len(np.unique(y)) <= min(50, int(np.sqrt(len(y))))

    def _validate_ann_regression(self, max_samples=1024):
        X_full, y_full = self.X_, self.y_
        n = X_full.shape[0]

        if n > max_samples:
            sel = np.random.choice(n, max_samples, replace=False)
            X = X_full[sel]
            y = y_full[sel]
        else:
            X, y = X_full, y_full

        ann = AnnBackend(X, use_gpu=False, silent=True)
        idx_mat, dist_mat = ann.kneighbors_batch(X, self.k)

        w = 1.0 / np.maximum(dist_mat, 1e-9)
        y_neighbors = y[idx_mat]
        preds = np.sum(y_neighbors * w, axis=1) / np.sum(w, axis=1)

        return r2_score(y, preds)

    def fit(self, X, y):
        self._validate_schema_array(X, y)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)

        with self._lock:
            self.fitted = False

            X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
            X_norm, self.mean_, self.std_ = normalize(X)

            w = learn_feature_weights(X_norm, y, self.alpha, self.beta, self.gamma)
            w = clip_weights(w)

            X_f, w_f, mask = filter_low_weights(
                X_norm, w, self.weight_threshold, return_mask=True
            )

            self.feature_mask_ = mask
            self.weights_ = w_f.astype(np.float32)
            self.X_ = X_f.astype(np.float32)
            self.y_ = y
            self.n_features_ = X_f.shape[1]

            self.is_classification_ = self._infer_classification(y)
            self.classes_ = np.unique(y) if self.is_classification_ else None

            backend_logger = logger.getChild("Backend")

            MIN_SAMPLES_FOR_ANN = 10_000

            if (
                self.backend_mode == "brute"
                or not ANN_AVAILABLE
                or self.X_.shape[0] < MIN_SAMPLES_FOR_ANN
            ):
                backend_logger.info(
                    f"Using BRUTE backend (samples={self.X_.shape[0]} — SMALL DATASET OR ANN UNAVAILABLE)."
                )
                self.backend = BruteBackend(self.X_, self.weights_)
            else:
                self.backend = AnnBackend(self.X_, use_gpu=self.use_gpu)

                if self.ann_quality_check and not self.is_classification_:
                    r2 = self._validate_ann_regression()
                    if r2 < self.ann_min_r2:
                        backend_logger.warning(
                            f"ANN quality failed (R²={r2:.3f}) — switching to BRUTE."
                        )
                        self.backend = BruteBackend(self.X_, self.weights_)
                    else:
                        backend_logger.info(f"ANN quality passed (R²={r2:.3f}).")

            self.fitted = True

        return self

    def _kneighbors_batch(self, Xq):
        Xq = np.asarray(Xq, dtype=np.float32)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)

        if not np.isfinite(Xq).all():
            warnings.warn(
                "NaN/Inf detected in query — APPLYING SAFE NORMALIZATION.",
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

    def predict(self, X):
        if not getattr(self, "fitted", False):
            raise RuntimeError("SmartKNN instance is not fitted yet.")

        idx, dist = self._kneighbors_batch(X)

        w = 1.0 / np.maximum(dist, 1e-9)
        y_neighbors = self.y_[idx]

        if self.is_classification_:
            classes = self.classes_
            class_idx = np.searchsorted(classes, y_neighbors)
            scores = np.zeros((idx.shape[0], len(classes)), dtype=np.float32)
            np.add.at(
                scores,
                (np.repeat(np.arange(idx.shape[0]), self.k), class_idx.ravel()),
                w.ravel()
            )
            return classes[np.argmax(scores, axis=1)]

        return np.sum(y_neighbors * w, axis=1) / np.sum(w, axis=1)

    def __getstate__(self):
        state = {slot: getattr(self, slot, None) for slot in self.__slots__}
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self._lock = threading.Lock()
