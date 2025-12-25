import numpy as np


def filter_low_weights(
    X,
    weights,
    threshold=0.0,
    min_features=1,
    return_mask=False
):
 
    if hasattr(X, "values"):
        X = X.values
    X = np.nan_to_num(np.asarray(X, dtype=np.float32, order="C"),
                      nan=0.0, posinf=1e9, neginf=-1e9)
    weights = np.asarray(weights, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if weights.shape[0] != X.shape[1]:
        raise ValueError(
            f"weights length {weights.shape[0]} != number of features {X.shape[1]}"
        )


    w = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None).astype(np.float32)


    if np.all(w == 0.0):
       
        sorted_idx = np.arange(len(w))      
        mask = np.zeros_like(w, dtype=bool)
        mask[sorted_idx[:min_features]] = True

        X_f = X[:, mask]
        w_f = w[mask]
        if return_mask:
            return X_f.astype(np.float32), w_f.astype(np.float32), mask
        return X_f.astype(np.float32), w_f.astype(np.float32)


    if threshold <= 0:
        mask = w > 0.0
    else:
        mask = w >= float(threshold)


    if mask.sum() < min_features:
        sorted_idx = np.argsort(w)[::-1]
        top_idx = sorted_idx[:min_features]
        mask = np.zeros_like(w, dtype=bool)
        mask[top_idx] = True


    X_f = X[:, mask]
    w_f = w[mask]

    if X_f.shape[1] == 0:
        raise RuntimeError(
            "Filtered out ALL features — lower threshold or adjust weights."
        )

    if return_mask:
        return X_f.astype(np.float32), w_f.astype(np.float32), mask
    return X_f.astype(np.float32), w_f.astype(np.float32)
