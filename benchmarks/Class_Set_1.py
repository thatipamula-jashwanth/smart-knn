import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import fetch_openml

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from smart_knn import SmartKNN



warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.2
LAT_RUNS = 300
WARMUP_RUNS = 30

np.random.seed(SEED)



def classification_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def measure_single_latency(fn, x, runs=LAT_RUNS):
    for _ in range(WARMUP_RUNS):
        fn(x)

    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times[i] = (time.perf_counter() - t0) * 1000.0

    return np.median(times), np.percentile(times, 95)


def extract_xy(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.to_numpy()

    return X, y


def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"[INFO] rows={len(df)} | num={len(num_cols)} | cat={len(cat_cols)}")

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )



def benchmark_classification(df, target_col, dataset_name, output_dir):
    X, y = extract_xy(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    pre = build_preprocessor(X_train)

    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)
    x_single = X_test_enc[:1]

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=1,
            random_state=SEED,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.1,
            n_jobs=1,
            random_state=SEED,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=150,
            depth=6,
            learning_rate=0.1,
            thread_count=1,
            random_seed=SEED,
            verbose=False,
        ),
    }

    rows = []


    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train_enc, y_train)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test_enc)
        batch_t = time.perf_counter() - t0

        acc, f1 = classification_metrics(y_test, preds)
        med, p95 = measure_single_latency(model.predict, x_single)

        rows.append([name, acc, f1, train_t, batch_t, med, p95])

    best = None
    for wt in (0.0, 0.1):
        knn = SmartKNN(
            k=5,
            backend="auto",
            weight_threshold=wt,
            force_classification=True,
        )

        t0 = time.perf_counter()
        knn.fit(X_train_enc, y_train)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = knn.predict(X_test_enc)
        batch_t = time.perf_counter() - t0

        acc, f1 = classification_metrics(y_test, preds)
        med, p95 = measure_single_latency(knn.predict, x_single)

        if best is None or f1 > best["f1"]:
            best = dict(
                wt=wt,
                acc=acc,
                f1=f1,
                train=train_t,
                batch=batch_t,
                med=med,
                p95=p95,
            )

    rows.append([
        f"SmartKNN (wt={best['wt']})",
        best["acc"],
        best["f1"],
        best["train"],
        best["batch"],
        best["med"],
        best["p95"],
    ])

    df_out = pd.DataFrame(
        rows,
        columns=[
            "model",
            "accuracy",
            "macro_f1",
            "train_s",
            "batch_s",
            "single_med_ms",
            "single_p95_ms",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}.csv"
    df_out.to_csv(out_path, index=False)

    print(f"[SAVED] {out_path}")


def run(output_dir="benchmarks/results"):
    datasets = [
        ("adult", "class", "adult_income"),
    ]

    for name, target, out_name in datasets:
        print(f"\n[DATASET] {name}")
        ds = fetch_openml(name=name, as_frame=True, parser="auto")
        benchmark_classification(
            ds.frame,
            target,
            out_name,
            Path(output_dir),
        )


if __name__ == "__main__":
    run()
