#  SmartKNN

A **smarter, weighted, feature-selective KNN algorithm** that automatically learns feature importance, filters weak features, handles missing values, normalizes data, and delivers significantly better accuracy than classical KNN — all with a **simple sklearn-like API**.

SmartKNN supports both **classification** and **regression**, requires zero manual tuning for preprocessing, and is fully compatible with **NumPy** and **Pandas**.

---

#  Badges

[![PyPI](https://img.shields.io/pypi/v/smart-knn)](https://pypi.org/project/smart-knn/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Stable-success)](#)
[![Downloads](https://img.shields.io/pypi/dm/smart-knn?label=Downloads)](https://pypi.org/project/smart-knn/)

---

#  Features

* **Automatic Feature Weighting**

  * Univariate MSE scoring
  * Mutual Information
  * Random Forest importance

* **Automatic Preprocessing**

  * Normalization
  * NaN / Inf cleaning
  * Median imputation
  * Value clipping

* **Automatic Feature Filtering**

  * Removes low-weight & noisy features
  * Keeps only important signals

* **Weighted Euclidean Distance**

* **Scikit-Learn Style API**

  * `fit()`
  * `predict()`
  * `kneighbors()`

* **Supports**

  * NumPy arrays
  * Pandas DataFrames
  * Regression + Classification

---

#  Installation

### Install from PyPI

`bash\	pip install smart-knn`

### Local install

```bash
pip install .
```

---

#  Quick Start

```python
import pandas as pd
from smart_knn import SmartKNN

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = SmartKNN(k=5)
model.fit(X, y)

sample = X.iloc[0]
pred = model.predict(sample)
print("Prediction:", pred)
```

SmartKNN will automatically:

* Normalize inputs
* Learn weights
* Clean NaN/Inf
* Filter weak features

---

#  Predict Multiple Rows

```python
preds = model.predict(X.iloc[:10])
print(preds)
```

---

#  How SmartKNN Works

1. Learns feature importance (MSE + MI + Random Forest).
2. Removes weak features.
3. Normalizes input.
4. Applies weighted Euclidean distance.
5. Optimized vectorized NumPy inference.

**Results:**

* Higher accuracy
* Faster prediction
* Lower noise sensitivity
* Better generalization

---

#  API Overview

### Initialize

```python
model = SmartKNN(k=5, weight_threshold=0.05)
```

### Fit

```python
model.fit(X, y)
```

### Predict

```python
model.predict(sample)
```

### Neighbors

```python
idx, dists = model.kneighbors(sample)
```

### Inspect Model

```python
model.weights_
model.feature_mask_
model.X_.shape
```

---
| Parameter            | Description                | Range |
| -------------------- | -------------------------- | ----- |
| **k**                | Number of neighbors        | 3–15  |
| **weight_threshold** | Drop features below weight | 0–0.3 |
| **n_jobs**           | Parallel workers (future)  | 1–8   |

(alpha/beta/gamma removed — not supported in current PyPI version)
---

# 📁 Project Structure

```
smart_knn/
 ├── base_knn.py
 ├── distance.py
 ├── weight_learning.py
 ├── data_processing.py
 ├── utils.py
 ├── evaluation.py
 ├── adaptive_k.py
 ├── prototypes.py
 └── signatures.py

docs/
 ├── design.md
 ├── theory.md
 ├── roadmap.md
 └── usage.md

benchmarks/
 ├── classification_tests/
 ├── regression_tests/
 └── heatmaps/
```

---

#  Benchmark Visuals

```
![Accuracy Heatmap](benchmarks/heatmaps/class_accuracy.png)
![Regression MSE](benchmarks/heatmaps/reg_mse.png)

```
---

#  Roadmap

* Adaptive-K
* Prototype compression
* Neural metric learning
* FAISS / HNSW accelerated search
* GPU support
* Distance signatures
* Incremental learning

---

#  License

SmartKNN is released under the **MIT License**.
See the `LICENSE` file for details.

---

#  Contributing

PRs and feature requests are welcome!
If you like SmartKNN, star  the repository.

---

# 🔗 Links

* **PyPI:** [https://pypi.org/project/smart-knn](https://pypi.org/project/smart-knn)
* **GitHub:** [https://github.com/thatipamula-jashwanth/smart-knn](https://github.com/thatipamula-jashwanth/smart-knn)
