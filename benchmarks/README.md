# SmartKNN Benchmarks

This directory contains **reproducible, engineering-focused benchmarks** for
**SmartKNN**, covering both **classification** and **regression** tasks.

The benchmarks are intended for **internal validation and analysis**, with an
emphasis on:

- predictive performance (accuracy / error)
- inference latency
- scalability characteristics

They are **not marketing benchmarks** and are not automatically executed.

---

## Benchmark Sets

### Classification
- **class_set_1.py** — SmartKNN vs gradient-boosted models (XGBoost, LightGBM, CatBoost)
- **class_set_2.py** — SmartKNN vs classical baselines (Logistic Regression, KNN, Trees)

### Regression
- **regression_set_1.py** — SmartKNN vs gradient-boosted models
- **regression_set_2.py** — SmartKNN vs classical regression baselines

---

## Running Benchmarks

From the project root:

```bash
python -m benchmarks.run_benchmark
```

---

## Running a Subset of Benchmarks

You can selectively run benchmarks using a substring filter:

```bash
python -m benchmarks.run_benchmark --pattern class
python -m benchmarks.run_benchmark --pattern regression
python -m benchmarks.run_benchmark --pattern set_1
python -m benchmarks.run_benchmark --pattern set_2
```

This allows targeted evaluation without modifying code.

---

## Results

All benchmark outputs are written to:

- benchmarks/results/

Results are stored as CSV files and are not committed to the repository.

These files are intended for:

- local inspection
- offline analysis
- report generation