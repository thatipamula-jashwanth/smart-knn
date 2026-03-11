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
except ImportError:
    ANN_AVAILABLE = False

from .smartknn_engine import (
    _validate_ann_regression,
    _kneighbors_batch
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SmartKNN:

    __slots__ = (
        "k","weight_threshold","alpha","beta","gamma",
        "force_classification",
        "backend_mode","use_gpu",
        "ann_quality_check","ann_min_r2",
        "ann_nlist","ann_nprobe",
        "global_lambda",
        "_lock","fitted",
        "mean_","std_",
        "impute_values_",
        "feature_mask_","X_","weights_","y_",
        "n_features_","n_features_in_",
        "is_classification_","classes_",
        "backend",
        "global_u_","X_global_"
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
        ann_min_r2=0.0,
        ann_nlist=None,
        ann_nprobe=None,
        global_lambda=0.20
    ):

        if k < 1:
            raise ValueError("k must be >= 1")

        if backend not in ("auto","brute","ann"):
            raise ValueError("Backend must be 'auto','brute','ann'")

        for name,val in {
            "weight_threshold":weight_threshold,
            "alpha":alpha,
            "beta":beta,
            "gamma":gamma,
            "ann_min_r2":ann_min_r2,
            "global_lambda":global_lambda
        }.items():
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")

        self.k=int(k)
        self.weight_threshold=float(weight_threshold)

        self.alpha=float(alpha)
        self.beta=float(beta)
        self.gamma=float(gamma)

        self.force_classification=force_classification

        self.backend_mode=backend
        self.use_gpu=bool(use_gpu)

        self.ann_quality_check=bool(ann_quality_check)
        self.ann_min_r2=float(ann_min_r2)

        self.ann_nlist=ann_nlist
        self.ann_nprobe=ann_nprobe

        self.global_lambda=float(global_lambda)

        self.fitted=False
        self._lock=threading.Lock()

    def _get_tags(self):
        return {"requires_y": True}

    def __sklearn_is_fitted__(self):

        required=("X_","y_","weights_","mean_","std_","feature_mask_","backend")

        return (
            getattr(self,"fitted",False)
            and all(hasattr(self,a) and getattr(self,a) is not None for a in required)
        )

    def get_params(self,deep=True):

        return {
            "k":self.k,
            "weight_threshold":self.weight_threshold,
            "alpha":self.alpha,
            "beta":self.beta,
            "gamma":self.gamma,
            "force_classification":self.force_classification,
            "backend":self.backend_mode,
            "use_gpu":self.use_gpu,
            "ann_quality_check":self.ann_quality_check,
            "ann_min_r2":self.ann_min_r2,
            "ann_nlist":self.ann_nlist,
            "ann_nprobe":self.ann_nprobe,
            "global_lambda":self.global_lambda
        }

    def set_params(self,**params):

        valid=self.get_params().keys()

        for key,value in params.items():

            if key not in valid:
                raise ValueError(f"Invalid parameter {key}")

            if key=="backend":
                self.backend_mode=value
            else:
                setattr(self,key,value)

        return self

    def _validate_schema_array(self,X,y=None):

        X=np.asarray(X)

        if X.ndim!=2:
            raise ValueError("X must be 2D.")

        if y is not None:
            y=np.asarray(y)
            if y.shape[0]!=X.shape[0]:
                raise ValueError("X and y length mismatch")

        return X,y

    def _infer_classification(self,y):

        if self.force_classification is True:
            return True

        if self.force_classification is False:
            return False

        if y.dtype == bool:
            return True

        if y.dtype.kind in ("U","S","O"):
            return True

        if np.issubdtype(y.dtype,np.integer):
            return len(np.unique(y)) < 100

        return False

    def _replace_nonfinite(self,X):

        finite=np.isfinite(X)

        X_tmp=X.copy()
        X_tmp[~finite]=np.nan

        med=np.nanmedian(X_tmp,axis=0)
        med=np.where(np.isfinite(med),med,0.0).astype(np.float32)

        X=X.copy()

        mask=~finite
        if mask.any():
            X[mask]=med[np.where(mask)[1]]

        return X,med

    def fit(self,X,y):

        X,y=self._validate_schema_array(X,y)

        X=np.asarray(X,dtype=np.float32)
        y=np.asarray(y).reshape(-1)

        if self.k>X.shape[0]:
            raise ValueError("k cannot exceed sample count")

        with self._lock:

            self.fitted=False

            X,med=self._replace_nonfinite(X)
            self.impute_values_=med

            X_norm,self.mean_,self.std_=normalize(X)

            self.std_=np.maximum(self.std_,1e-12)

            self.n_features_in_=X.shape[1]

            w=learn_feature_weights(X_norm,y,self.alpha,self.beta,self.gamma)
            w=clip_weights(w)

            X_f,w_f,mask=filter_low_weights(
                X_norm,w,self.weight_threshold,return_mask=True
            )

            if X_f.shape[1]==0:
                raise ValueError("All features removed by weight_threshold")

            self.feature_mask_=mask
            self.weights_=w_f.astype(np.float32)

            self.X_=X_f.astype(np.float32)
            self.y_=y

            self.n_features_=X_f.shape[1]

            self.is_classification_=self._infer_classification(y)

            if self.is_classification_:
                self.classes_=np.unique(y)
            else:
                self.classes_=None

            try:

                if not self.is_classification_:

                    r=min(3,self.X_.shape[1])

                    A=self.X_.T@(self.y_[:,None]*self.X_)

                    eigvals,eigvecs=np.linalg.eigh(A)

                    idx=np.argsort(eigvals)[::-1][:r]

                    U=eigvecs[:,idx]

                    U=U/np.maximum(np.linalg.norm(U,axis=0,keepdims=True),1e-12)

                    self.global_u_=U.astype(np.float32)
                    self.X_global_=self.X_@self.global_u_

                else:

                    self.global_u_=None
                    self.X_global_=None

            except Exception as e:

                logger.debug(f"Global projection failed: {e}")

                self.global_u_=None
                self.X_global_=None

            backend_logger=logger.getChild("Backend")

            MIN_SAMPLES_FOR_ANN=10000

            if self.backend_mode=="brute":

                self.backend=BruteBackend(self.X_,self.weights_)

            elif self.backend_mode=="ann":

                if not ANN_AVAILABLE:
                    raise RuntimeError("ANN backend requested but unavailable")

                self.backend=AnnBackend(
                    self.X_,
                    nlist=self.ann_nlist,
                    nprobe=self.ann_nprobe,
                    use_gpu=self.use_gpu
                )

            else:

                if ANN_AVAILABLE and self.X_.shape[0]>=MIN_SAMPLES_FOR_ANN:

                    try:

                        self.backend=AnnBackend(
                            self.X_,
                            nlist=self.ann_nlist,
                            nprobe=self.ann_nprobe,
                            use_gpu=self.use_gpu
                        )

                    except Exception as e:

                        backend_logger.warning(
                            f"ANN init failed ({e}) -> Brute fallback"
                        )

                        self.backend=BruteBackend(self.X_,self.weights_)

                else:

                    self.backend=BruteBackend(self.X_,self.weights_)

            if self.ann_quality_check and not self.is_classification_ and isinstance(self.backend,AnnBackend):

                r2=_validate_ann_regression(self)

                if r2<self.ann_min_r2:

                    backend_logger.warning(
                        f"ANN quality failed (R²={r2:.3f}) → switching to BRUTE."
                    )

                    self.backend=BruteBackend(self.X_,self.weights_)

            self.fitted=True

        return self

    def _prepare_query(self,X):

        Xq=np.asarray(X,dtype=np.float32)

        if Xq.ndim==1:
            Xq=Xq.reshape(1,-1)

        if Xq.ndim!=2:
            raise ValueError("X must be 2D.")

        if Xq.shape[1]!=self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {Xq.shape[1]}"
            )

        mask=~np.isfinite(Xq)

        if mask.any():
            warnings.warn("NaN/Inf detected in query", RuntimeWarning)
            logger.debug("NaN/Inf detected in query batch")
            Xq[mask]=self.impute_values_[np.where(mask)[1]]

        Xq=(Xq-self.mean_)/self.std_

        Xq=Xq[:,self.feature_mask_]

        return Xq
        
    def _kneighbors_batch(self, X):
        if not self.fitted:
            raise RuntimeError("SmartKNN not fitted")

        Xq = self._prepare_query(X)

        return _kneighbors_batch(self, Xq)
    
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("SmartKNN not fitted")
        
        Xq = self._prepare_query(X)
        batch_size = 2000
        all_preds = []
        
        for i in range(0, Xq.shape[0], batch_size):
            
            batch = Xq[i:i + batch_size]
            
            idx, dist = _kneighbors_batch(self, batch)
            
            w = 1.0 / np.maximum(dist, 1e-9)
            
            y_neighbors = self.y_[idx]
            
            if self.is_classification_:
                class_idx = np.searchsorted(self.classes_, y_neighbors)
                
                scores = np.zeros((idx.shape[0], len(self.classes_)), dtype=np.float32)
                
                np.add.at(
                    scores,
                    (np.repeat(np.arange(idx.shape[0]), self.k), class_idx.ravel()),
                    w.ravel()
                )
                
                pred = self.classes_[np.argmax(scores, axis=1)]

            else:

                pred = np.sum(y_neighbors * w, axis=1) / np.sum(w, axis=1)

            all_preds.append(pred)

        return np.concatenate(all_preds)
    
    def predict_proba(self, X):
        
        if not self.is_classification_:
            raise RuntimeError("predict_proba only for classification")

        Xq = self._prepare_query(X)

        batch_size = 2000
        all_probs = []

        for i in range(0, Xq.shape[0], batch_size):

            batch = Xq[i:i + batch_size]

            idx, dist = _kneighbors_batch(self, batch)

            w = 1.0 / np.maximum(dist, 1e-9)

            y_neighbors = self.y_[idx]

            class_idx = np.searchsorted(self.classes_, y_neighbors)

            scores = np.zeros((idx.shape[0], len(self.classes_)), dtype=np.float32)

            np.add.at(
                scores,
                (np.repeat(np.arange(idx.shape[0]), self.k), class_idx.ravel()),
                w.ravel()
            )

            scores_sum = np.sum(scores, axis=1, keepdims=True)

            probs = scores / np.maximum(scores_sum, 1e-12)

            all_probs.append(probs)

        return np.vstack(all_probs)

    def decision_function(self,X):

        probs=self.predict_proba(X)

        if probs.shape[1]==2:
            return probs[:,1]

        return probs

    def score(self,X,y):

        y=np.asarray(y).reshape(-1)

        pred=self.predict(X)

        if pred.shape[0]!=y.shape[0]:
            raise ValueError("Prediction length mismatch")

        if self.is_classification_:
            return np.mean(pred==y)

        return r2_score(y,pred)