#  SmartKNN Benchmarks

This directory contains **reproducible benchmarks** for **SmartKNN**, covering
both **classification** and **regression** performance against common ML baselines.

Benchmarks focus on **accuracy, latency, and scalability**.
They are intended for **engineering validation**, not marketing claims.

---

##  Benchmark Sets

### Classification
- **class_set_1.py** — SmartKNN vs GBM models
- **class_set_2.py** — SmartKNN vs classical baselines

### Regression
- **regression_set_1.py** — SmartKNN vs GBM models
- **regression_set_2.py** — SmartKNN vs classical baselines

---

##  Running Benchmarks

From the project root:

```bash
python benchmarks/run_benchmarks.py
```

## Run a subset

Benchmarks are auto-discovered by filename.
You can filter runs using patterns:

python benchmarks/run_benchmarks.py --pattern class
python benchmarks/run_benchmarks.py --pattern regression
python benchmarks/run_benchmarks.py --pattern set_1
python benchmarks/run_benchmarks.py --pattern set_2

---

## Results

All benchmark outputs are written to:

benchmarks/results/

Results are not committed to the repository.

In CI, results are uploaded as workflow artifacts.

---

## Continuous Integration (CI)
Benchmarks are executed via GitHub Actions (benchmarks.yml) using:

- Manual workflow triggers
- Optional pattern-based filtering

Benchmark failures do not block pull requests.

---

## Adding New Benchmarks

1. Add a new file:
benchmarks/<type>_set_<n>.py

2. Implement:
def run(output_dir: str):
    ...
3. Write all results to output_dir
No CI changes are required.

---

**SmartKNN benchmarks are designed to be transparent, reproducible, and fair.**



