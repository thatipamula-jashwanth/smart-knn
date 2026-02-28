<p align="center">
  <img src="smart_knn/logo/SmartKNN.png" alt="SmartKNN logo" width="160" />
</p>

<h1 align="center">SmartKNN</h1>

<p align="center">
A modern, weighted nearest-neighbor learning algorithm with learned feature importance and adaptive neighbor search.
</p>

<p align="center">
  <a href="https://thatipamula-jashwanth.github.io/SmartEco/">
    <img src="https://img.shields.io/badge/Website-SmartEco-blue?style=flat-square" alt="Website"/>
  </a>
  <a href="https://pypi.org/project/smart-knn/">
    <img src="https://img.shields.io/pypi/v/smart-knn?style=flat-square" alt="PyPI version"/>
  </a>
  <a href="https://pypi.org/project/smart-knn/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square" alt="Python versions"/>
  </a>
  <a href="https://github.com/thatipamula-jashwanth/smart-knn/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/thatipamula-jashwanth/smart-knn/ci-tests.yml?style=flat-square" alt="CI status"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"/>
  </a>
  <a href="https://huggingface.co/spaces/JashuXo/smart-knn">
  <img src="https://img.shields.io/badge/HuggingFace-Demo-yellow?style=flat-square" alt="Hugging Face Demo"/>
</a>
</p>

## Overview

**SmartKNN** is a nearest-neighbor–based learning method that belongs to the
broader **KNN family** of algorithms.

It is designed to address common limitations observed in classical KNN
approaches, including:

- uniform treatment of all features  
- sensitivity to noisy or weakly informative dimensions  
- limited scalability as dataset size grows  

SmartKNN incorporates **data-driven feature importance estimation**,
**dimension suppression**, and **adaptive neighbor search strategies**.
Depending on dataset characteristics, it can operate using either a
brute-force search or an approximate nearest-neighbor (ANN) backend, while
exposing a consistent, **scikit-learn–compatible API**.

The method supports both **regression** and **classification** tasks and
prioritizes **robustness**, **predictive accuracy**, and **practical inference latency**
across a range of dataset sizes.

---

## Key Capabilities

- **Learned feature weighting**
  - MSE relevance
  - Mutual Information
  - Random Forest importance  
  *(method configurable depending on task and dataset)*
- **Automatic preprocessing**
  - normalization
  - NaN / Inf handling
  - feature masking
- **Distance-weighted neighbor voting**
- **Brute-force and ANN backends**
  - designed to scale to large datasets (hardware and tuning dependent)
  - optional GPU-accelerated neighbor search
- **Vectorized NumPy with Numba acceleration**
- **Scikit-learn–compatible API**

---

## Installation

```bash
pip install smart-knn
```

---

## Documentation

- https://thatipamula-jashwanth.github.io/SmartEco/

Detailed documentation and design notes are maintained externally.
This repository README is intentionally kept concise.

---
## Examples

Runnable examples are available in the **examples/ directory**:
```
python examples/regression_example.py
python examples/classification_example.py
```

---
## Benchmarks & CI

- Comprehensive benchmark suites for regression and classification
- GitHub Actions CI for tests and benchmarks
- Reproducible, engineering-focused evaluation

Benchmark details are documented in **benchmarks/README.md**.

---
## Project Status

- SmartKNN v2 is stable
- API is frozen for the v2.x series (backward-compatible improvements only)
- Actively maintained
- Open to research and engineering collaboration

---
## License

**SmartKNN** is released under the **MIT License**.
See **LICENSE** for details.
