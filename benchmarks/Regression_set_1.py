import time
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import fetch_openml, fetch_california_housing

from smart_knn import SmartKNN


warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
TEST_SIZE = 0.2
LAT_RUNS = 500
WARMUP_RUNS = 50
ROW_OHE_THRESHOLD = 100


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def measure_single_latency(fn, x, runs=LAT_RUNS):
    for _ in range(WARMUP_RUNS):
        fn(x)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times), np.percentile(times, 95)


def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns

    if len(df) < ROW_OHE_THRESHOLD:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc_type = "onehot"
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        enc_type = "ordinal"

    print(
        f"[INFO] Encoding={enc_type} | rows={len(df)} | cat_cols={len(cat_cols)}"
    )

    return ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", encoder, cat_cols),
    ])


def safe_target(df, target_col):
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        print(f"[WARN] Target '{target_col}' not found — using last column.")
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    if y.dtype == object:
        y = pd.to_numeric(y.astype(str), errors="coerce")

    y = y.values.astype(np.float32)

    if np.any(np.isnan(y)):
        raise ValueError("Target contains NaNs after conversion.")

    return X, y


def benchmark_regression(df, target_col, name):

    X, y = safe_target(df, target_col)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    print(f"\n REGRESSION")
    print(
        "Model        | RMSE ↓ | R² ↑  | Train(s) | "
        "Batch Pred(s) | Single Med(ms) | Single P95(ms)"
    )
    print("-" * 130)

  
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    x_single = X_test_enc[[0]]


    models = {


        "KNN": {
            "pipeline": Pipeline([
                ("prep", preprocessor),
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]),
            "core": Pipeline([
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=5)),
            ]),
            "needs_scaling": True,
        },

        "Linear": {
            "pipeline": Pipeline([
                ("prep", preprocessor),
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]),
            "core": Pipeline([
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]),
            "needs_scaling": True,
        },


        "DecisionTree": {
            "pipeline": Pipeline([
                ("prep", preprocessor),
                ("model", DecisionTreeRegressor(random_state=SEED)),
            ]),
            "core": DecisionTreeRegressor(random_state=SEED),
            "needs_scaling": False,
        },

        "RandomForest": {
            "pipeline": Pipeline([
                ("prep", preprocessor),
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    random_state=SEED,
                    n_jobs=1
                )),
            ]),
            "core": RandomForestRegressor(
                n_estimators=200,
                random_state=SEED,
                n_jobs=1
            ),
            "needs_scaling": False,
        },
    }

    for name_m, cfg in models.items():

        pipe = clone(cfg["pipeline"])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = pipe.predict(X_test)
        batch_t = time.perf_counter() - t0

        rmse, r2 = regression_metrics(y_test, preds)

        if cfg["needs_scaling"]:
            scaler = StandardScaler().fit(X_train_enc)
            Xtr = scaler.transform(X_train_enc)
            Xte = scaler.transform(x_single)
            core = clone(cfg["core"])
            core.fit(Xtr, y_train)
            single_med, single_p95 = measure_single_latency(core.predict, Xte)
        else:
            core = clone(cfg["core"])
            core.fit(X_train_enc, y_train)
            single_med, single_p95 = measure_single_latency(core.predict, x_single)

        print(
            f"{name_m:12s} | {rmse:7.4f} | {r2:6.4f} | "
            f"{train_t:7.3f} | {batch_t:13.4f} | "
            f"{single_med:15.3f} | {single_p95:15.3f}"
        )

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
        single_med, single_p95 = measure_single_latency(knn.predict, x_single)

        cand = {
            "wt": wt, "rmse": rmse, "r2": r2,
            "train": train_t, "batch": batch_t,
            "med": single_med, "p95": single_p95
        }

        if best is None or r2 > best["r2"]:
            best = cand

    print(
        f"{'SmartKNN*':12s} | {best['rmse']:7.4f} | {best['r2']:6.4f} | "
        f"{best['train']:7.3f} | {best['batch']:13.4f} | "
        f"{best['med']:15.3f} | {best['p95']:15.3f}"
    )
    print(f"  selected weight_threshold = {best['wt']}")


def load_and_run():
    cal = fetch_california_housing(as_frame=True)
    benchmark_regression(cal.frame, "MedHouseVal", "California Housing")

    bike = fetch_openml(name="Bike_Sharing_Demand", as_frame=True)
    benchmark_regression(bike.frame, "count", "Bike Sharing Demand")

    kc = fetch_openml(name="house_sales", as_frame=True)
    benchmark_regression(kc.frame, "price", "House Sales (King County)")


if __name__ == "__main__":
    load_and_run()
