import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def _ensure_numpy(x):
    if hasattr(x, "values"):
        x = x.values
    return np.asarray(x)


def _check_lengths(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true length {len(y_true)} != y_pred length {len(y_pred)}")


def _clean_numeric(y):
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def evaluate_regression(y_true, y_pred):
    y_true = _ensure_numpy(_clean_numeric(y_true)).astype(np.float32)
    y_pred = _ensure_numpy(_clean_numeric(y_pred)).astype(np.float32)

    _check_lengths(y_true, y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }



def _map_labels_vectorized(y_pred_raw, class_to_idx):
    classes = np.array(list(class_to_idx.keys()))
    indices = np.array(list(class_to_idx.values()))

    match = (y_pred_raw.reshape(-1, 1) == classes).argmax(axis=1)
    valid = (y_pred_raw.reshape(-1, 1) == classes).any(axis=1)
    result = indices[match]
    result[~valid] = -1      
    return result.astype(int)


def evaluate_classification(y_true, y_pred):
    y_true_raw = _ensure_numpy(y_true)
    y_pred_raw = _ensure_numpy(y_pred)

    if y_true_raw.dtype.kind not in "iu":
        classes, y_true_enc = np.unique(y_true_raw, return_inverse=True)
        class_to_idx = {c: i for i, c in enumerate(classes)}


        y_pred_enc = _map_labels_vectorized(y_pred_raw, class_to_idx)

        y_true = y_true_enc
        y_pred = y_pred_enc


    else:
        y_true = y_true_raw.astype(int)
        if y_pred_raw.dtype.kind in "fc":
            y_pred = np.rint(y_pred_raw).astype(int)
        else:
            y_pred = y_pred_raw.astype(int)

    _check_lengths(y_true, y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).astype(int)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def evaluate_auto(y_true, y_pred):
    y_np = _ensure_numpy(y_true)
    kind = y_np.dtype.kind

    if kind in ("O", "U", "S"):
        return evaluate_classification(y_true, y_pred)

    if kind in ("i", "u"):
        if len(np.unique(y_np)) > 50:
            return evaluate_regression(y_true, y_pred)
        return evaluate_classification(y_true, y_pred)

    if kind == "f":
        return evaluate_regression(y_true, y_pred)

    return evaluate_classification(y_true, y_pred)


def print_regression_report(metrics):
    print("\nRegression Report:")
    print(f"MSE  : {metrics['mse']:.4f}")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"R²   : {metrics['r2']:.4f}")


def print_classification_report(metrics):
    print("\nClassification Report:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])
