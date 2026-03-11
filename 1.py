import warnings
import tracemalloc
import numpy as np
import pandas as pd
import time

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from category_encoders import TargetEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from smart_knn import SmartKNN

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.2
N_JOBS = -1
N_SPLITS = 3

RUNS = 3
WARMUP = 1

BATCH_SIZES = [32, 256, 4096]
np.random.seed(SEED)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _latency_ms(fn, x):
    for _ in range(WARMUP):
        fn(x)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(times)), float(np.percentile(times, 95))


def _memory_peak_fit_mb(fit_fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    fit_fn()
    train_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return train_s, peak / (1024 * 1024)


def _safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) > 2:
            return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
        arr = np.asarray(scores)
        if arr.ndim == 2:
            return float(roc_auc_score(y_true, arr[:, 1]))
        return float(roc_auc_score(y_true, arr))
    except:
        return np.nan


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _openml_load(openml_id, target_name):
    from openml import datasets as oml_datasets
    ds = oml_datasets.get_dataset(openml_id)
    df, *_ = ds.get_data(dataset_format="dataframe")
    if target_name is None:
        target_name = ds.default_target_attribute
    return df, target_name


def _split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = y.astype("category").cat.codes

    return X, y.astype(np.int32)


def oof_target_encode(X_train, y_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imp.transform(X_test[num_cols])

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("missing")
        X_test[c] = X_test[c].astype(str).fillna("missing")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    X_oof = pd.DataFrame(index=X_train.index)
    for col in X_train.columns:
        X_oof[col] = 0.0

    for tr_idx, val_idx in kf.split(X_train, y_train):
        enc = TargetEncoder(cols=cat_cols, smoothing=0.3)
        enc.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        X_oof.iloc[val_idx] = enc.transform(X_train.iloc[val_idx])
        
    enc_full = TargetEncoder(cols=cat_cols, smoothing=0.3)
    enc_full.fit(X_train, y_train)
    X_test_enc = enc_full.transform(X_test)

    X_oof = X_oof.fillna(0).astype(np.float32)
    X_test_enc = X_test_enc.fillna(0).astype(np.float32)

    return X_oof, X_test_enc


def model_zoo():
    return [
        ("SmartKNN", lambda: SmartKNN(k=5, force_classification=True)),
        ("LightGBM", lambda: LGBMClassifier(random_state=SEED, n_jobs=N_JOBS)),
        ("CatBoost", lambda: CatBoostClassifier(random_seed=SEED, verbose=False)),
        ("XGBoost", lambda: XGBClassifier(
            random_state=SEED,
            n_jobs=N_JOBS,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )),
        ("DecisionTree", lambda: DecisionTreeClassifier(random_state=SEED)),
        ("LogReg", lambda: LogisticRegression(max_iter=1000)),
    ]


def benchmark_dataset(openml_id, name, target, out_dir):

    log(f"\n {name}")

    df, target_col = _openml_load(openml_id, target)
    X, y = _split_xy(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    X_train_enc, X_test_enc = oof_target_encode(X_train, y_train, X_test)

    rows = []

    for model_name, make_model in model_zoo():

        log(f"[CV] {model_name}")

        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

        cv_f1, cv_acc, cv_auc = [], [], []

        for tr_idx, val_idx in kf.split(X_train_enc, y_train):

            Xtr = X_train_enc.iloc[tr_idx].to_numpy()
            Xval = X_train_enc.iloc[val_idx].to_numpy()
            ytr = y_train.iloc[tr_idx].to_numpy()
            yval = y_train.iloc[val_idx].to_numpy()

            model = make_model()

            if model_name == "LogReg":
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xval = scaler.transform(Xval)

            model.fit(Xtr, ytr)
            y_pred = model.predict(Xval)
            scores = _get_scores(model, Xval)

            cv_f1.append(f1_score(yval, y_pred, average="macro"))
            cv_acc.append(accuracy_score(yval, y_pred))
            cv_auc.append(_safe_auc(yval, scores))

        model = make_model()

        Xtr_full = X_train_enc.to_numpy()
        Xte_full = X_test_enc.to_numpy()

        if model_name == "LogReg":
            scaler = StandardScaler()
            Xtr_full = scaler.fit_transform(Xtr_full)
            Xte_full = scaler.transform(Xte_full)

        train_s, mem_mb = _memory_peak_fit_mb(lambda: model.fit(Xtr_full, y_train))

        y_pred = model.predict(Xte_full)
        scores = _get_scores(model, Xte_full)

        test_f1 = f1_score(y_test, y_pred, average="macro")
        test_acc = accuracy_score(y_test, y_pred)
        test_auc = _safe_auc(y_test, scores)
        single_med, single_p95 = _latency_ms(model.predict, Xte_full[:1])

        rows.append({
            "model": model_name,
            "cv_f1_mean": np.mean(cv_f1),
            "cv_f1_std": np.std(cv_f1),
            "cv_acc_mean": np.mean(cv_acc),
            "cv_auc_mean": np.nanmean(cv_auc),
            "test_f1": test_f1,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "train_s": train_s,
            "fit_peak_mem_mb": mem_mb,
            "single_med_ms": single_med,
            "single_p95_ms": single_p95,
        })

    df_out = pd.DataFrame(rows).sort_values("test_f1", ascending=False)

    print("\nRESULTS")
    print(df_out.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / f"{name}.csv", index=False)


def run():
    datasets = [
        (45068, "Adult", "class"),
        (42477, "CreditCardDefault", "y"),
        (45566, "SantanderCustomerSatisfaction", "target"),
        (42397, "CreditCardFraudDetection", "Class"),
        (43948, "Covertype", "class"),
    ]

    out = Path("results_classification")
    for did, name, target in datasets:
        benchmark_dataset(did, name, target, out)


if __name__ == "__main__":
    run()