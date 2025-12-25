import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

def _safe_normalize(w, eps=1e-8):
    w = np.asarray(w, dtype=np.float32)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, eps, None)
    s = np.sum(w)
    return (w / (s + eps)).astype(np.float32)


def _univariate_mse_weights(X, y, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    var = np.var(X, axis=0)
    mask = var >= 1e-12

    n_features = X.shape[1]
    mse = np.full(n_features, 1.0, dtype=np.float32)
    if not mask.any():
        return np.ones(n_features, dtype=np.float32) / n_features

    Xm = X[:, mask]
    y_mean = y.mean()
    Xm_mean = Xm.mean(axis=0)
    y_c = y - y_mean
    Xm_c = Xm - Xm_mean

    cov = np.mean(Xm_c * y_c[:, None], axis=0)
    slope = cov / np.maximum(var[mask], eps)
    intercept = y_mean - slope * Xm_mean

    pred = Xm * slope + intercept
    mse_vals = np.mean((y[:, None] - pred) ** 2, axis=0)
    mse[mask] = mse_vals
    return _safe_normalize(1 / (mse + eps), eps)


def _fast_mi_weights(X, y, bins=32, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    N = len(X)
    if N > 50000:
        idx = np.random.choice(N, 50000, replace=False)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs, ys = X, y

    edges = np.percentile(ys, np.linspace(0, 100, bins + 1))
    yb = np.digitize(ys, edges) - 1

    mi = np.zeros(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        x = Xs[:, j]
        edges = np.percentile(x, np.linspace(0, 100, bins + 1))
        xb = np.digitize(x, edges) - 1

    
        joint = np.histogram2d(xb, yb, bins=bins)[0]
        pxy = joint / np.sum(joint)
        px = np.sum(pxy, axis=1, keepdims=True)
        py = np.sum(pxy, axis=0, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log((pxy + eps) / (px @ py + eps))
            mi[j] = np.sum(pxy * log_term)

    return _safe_normalize(mi + eps, eps)


def _fast_rf_weights(X, y, n_estimators=200, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    N = len(X)
    if N > 80000:
        idx = np.random.choice(N, 80000, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    try:
        rf = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            bootstrap=False,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(Xs, ys)
        imp = np.nan_to_num(rf.feature_importances_, nan=0.0)
        return _safe_normalize(imp + eps, eps)
    except Exception:
        return np.ones(X.shape[1], dtype=np.float32) / X.shape[1]


def learn_feature_weights(X, y, alpha=0.4, beta=0.3, gamma=0.3, eps=1e-8):
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)
    y = np.asarray(y, dtype=np.float32)

    w_mse = _univariate_mse_weights(X, y, eps=eps)
    w_mi = _fast_mi_weights(X, y, eps=eps)
    w_rf = _fast_rf_weights(X, y, eps=eps)

    weights = alpha * w_mse + beta * w_mi + gamma * w_rf
    return _safe_normalize(weights, eps)
