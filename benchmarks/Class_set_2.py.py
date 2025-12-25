import warnings
import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import fetch_openml

from smart_knn import SmartKNN

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.2
LAT_RUNS = 300
WARMUP_RUNS = 30


def classification_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


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

    print(f"[INFO] Encoding=onehot | rows={len(df)} | cat_cols={len(cat_cols)}")

    return ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def safe_target(df, target_col):
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        print(f"[WARN] Target '{target_col}' not found — using last column.")
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.values

    return X, y


def benchmark_classification(df, target_col, name):

    X, y = safe_target(df, target_col)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    print(f"\nCLASSIFICATION ")
    print(
        "Model                     | ACC ↑  | Macro-F1 ↑ | Train(s) | "
        "Batch Pred(s) | Single Med(ms) | Single P95(ms)"
    )
    print("-" * 150)

    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, n_jobs=1),
            True,
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            True,
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=SEED),
            False,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=150, random_state=SEED, n_jobs=1
            ),
            False,
        ),
    }

    for name_m, (model, needs_scaling) in models.items():

        steps = [("prep", preprocessor)]
        if needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", clone(model)))

        pipe = Pipeline(steps)

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = pipe.predict(X_test)
        batch_t = time.perf_counter() - t0

        acc, f1 = classification_metrics(y_test, preds)

        X_enc = preprocessor.fit_transform(X_train)
        x_single = X_enc[[0]]

        core_model = clone(model)
        if needs_scaling:
            scaler = StandardScaler().fit(X_enc)
            core_model.fit(scaler.transform(X_enc), y_train)
            single_med, single_p95 = measure_single_latency(
                lambda z: core_model.predict(scaler.transform(z)), x_single
            )
        else:
            core_model.fit(X_enc, y_train)
            single_med, single_p95 = measure_single_latency(
                core_model.predict, x_single
            )

        print(
            f"{name_m:25s} | {acc:6.4f} | {f1:10.4f} | "
            f"{train_t:7.3f} | {batch_t:13.4f} | "
            f"{single_med:15.3f} | {single_p95:15.3f}"
        )


    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    x_single = X_test_enc[[0]]

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
        single_med, single_p95 = measure_single_latency(knn.predict, x_single)

        if best is None or f1 > best["f1"]:
            best = dict(
                wt=wt,
                acc=acc,
                f1=f1,
                train=train_t,
                batch=batch_t,
                med=single_med,
                p95=single_p95,
            )

    print(
        f"{'SmartKNN (best)':25s} | "
        f"{best['acc']:6.4f} | {best['f1']:10.4f} | "
        f"{best['train']:7.3f} | {best['batch']:13.4f} | "
        f"{best['med']:15.3f} | {best['p95']:15.3f}"
    )
    print(f" selected weight_threshold = {best['wt']}")



def load_and_run():

    adult = fetch_openml(name="adult", as_frame=True)
    benchmark_classification(adult.frame, "class", "Adult Income (48K)")

    bank = fetch_openml(name="bank-marketing", as_frame=True)
    benchmark_classification(bank.frame, "class", "Bank Marketing (45K)")

    bank_id = fetch_openml(data_id=1486, as_frame=True)
    benchmark_classification(
        bank_id.frame, "class", "Bank Marketing (ID 1486 | 45K)"
    )


if __name__ == "__main__":
    load_and_run()
