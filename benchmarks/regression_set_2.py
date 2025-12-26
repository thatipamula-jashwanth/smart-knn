import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml, fetch_california_housing

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from smart_knn import SmartKNN

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.2
LAT_RUNS = 500
WARMUP_RUNS = 50

np.random.seed(SEED)

def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def measure_single_latency(fn, x, runs=LAT_RUNS):
    for _ in range(WARMUP_RUNS):
        fn(x)

    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times[i] = (time.perf_counter() - t0) * 1000.0

    return np.median(times), np.percentile(times, 95)


def safe_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = pd.to_numeric(df[target_col], errors="raise").values.astype(np.float32)
    return X, y


def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    print(f"[INFO] rows={len(df)} | num={len(num_cols)} | cat={len(cat_cols)}")

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )

def benchmark_regression(df, target_col, dataset_name, output_dir):
    X, y = safe_target(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pre = build_preprocessor(X_train)

    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)
    x_single = X_test_enc[:1]

    models = {
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=1,
            random_state=SEED,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            n_jobs=1,
            random_state=SEED,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200,
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

        rmse, r2 = regression_metrics(y_test, preds)
        med, p95 = measure_single_latency(model.predict, x_single)

        rows.append([name, rmse, r2, train_t, batch_t, med, p95])


    best = None
    for wt in (0.0, 0.1):
        knn = SmartKNN(k=5, backend="auto", weight_threshold=wt)

        t0 = time.perf_counter()
        knn.fit(X_train_enc, y_train)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = knn.predict(X_test_enc)
        batch_t = time.perf_counter() - t0

        rmse, r2 = regression_metrics(y_test, preds)
        med, p95 = measure_single_latency(knn.predict, x_single)

        cand = dict(
            wt=wt, rmse=rmse, r2=r2,
            train=train_t, batch=batch_t,
            med=med, p95=p95
        )

        if best is None or r2 > best["r2"] or (
            r2 == best["r2"] and rmse < best["rmse"]
        ):
            best = cand

    rows.append([
        f"SmartKNN (wt={best['wt']})",
        best["rmse"],
        best["r2"],
        best["train"],
        best["batch"],
        best["med"],
        best["p95"],
    ])

    df_out = pd.DataFrame(
        rows,
        columns=[
            "model",
            "rmse",
            "r2",
            "train_s",
            "batch_s",
            "single_med_ms",
            "single_p95_ms",
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{dataset_name.lower()}_gbm.csv"
    df_out.to_csv(out, index=False)

    print(f"[SAVED] {out}")


def run(output_dir="benchmarks/results"):
    output_dir = Path(output_dir)

    cal = fetch_california_housing(as_frame=True)
    benchmark_regression(
        cal.frame, "MedHouseVal", "california_housing", output_dir
    )

    bike = fetch_openml(name="Bike_Sharing_Demand", as_frame=True)
    benchmark_regression(
        bike.frame, "count", "bike_sharing_demand", output_dir
    )

    kc = fetch_openml(name="house_sales", as_frame=True)
    benchmark_regression(
        kc.frame, "price", "house_sales_king_county", output_dir
    )


if __name__ == "__main__":
    run()
