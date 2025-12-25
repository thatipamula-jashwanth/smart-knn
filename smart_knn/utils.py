import numpy as np


def ensure_numpy(X):
   
    if hasattr(X, "values"):
        X = X.values
    X = np.asarray(X, dtype=np.float32, order="C")
    return np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)


def ensure_2d(X):
    X = ensure_numpy(X)
    if X.ndim == 1:
        return X.reshape(1, -1)
    return X


def sanitize_input(X):

    return np.nan_to_num(np.asarray(X, dtype=np.float32, order="C"),
                         nan=0.0, posinf=1e9, neginf=-1e9)


def safe_std(X, eps=1e-8):
   
    s = np.std(X, axis=0)
    s = np.where(s < eps, eps, s)
    return s.astype(np.float32)


def normalize(X, eps=1e-8):
    X = sanitize_input(X)

 
    mean = X.mean(axis=0, dtype=np.float32)
    std = np.std(X, axis=0, dtype=np.float32)
    std = np.where(std < eps, eps, std).astype(np.float32)


    X_norm = (X - mean) / std
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return X_norm.astype(np.float32), mean, std


def apply_normalization(X, mean, std, eps=1e-8):
    X = sanitize_input(X)

    std_safe = np.where(std < eps, eps, std).astype(np.float32)
    Xn = (X - mean) / std_safe
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    return Xn.astype(np.float32)


def clip_weights(w, min_val=1e-6):
    w = np.nan_to_num(np.asarray(w, dtype=np.float32),
                      nan=min_val, posinf=min_val, neginf=min_val)
    return np.where(w < min_val, min_val, w).astype(np.float32)


def safe_normalize_vector(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    s = float(np.sum(v))
    if s < eps:
        return (np.ones_like(v) / len(v)).astype(np.float32)
    return (v / (s + eps)).astype(np.float32)


def ensure_feature_mask(X, mask):
    if mask is None:
        return np.ones(X.shape[1], dtype=bool)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != X.shape[1]:
        raise ValueError(
            f"Feature mask length {mask.shape[0]} != number of features {X.shape[1]}"
        )
    if not mask.any():
        raise ValueError("Feature mask removed ALL features. At least one feature must remain.")
    return mask
