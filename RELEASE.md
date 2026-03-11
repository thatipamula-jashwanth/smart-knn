# SmartKNN Release Notes

---

# Release 0.2.3 - **SmartKNN v2.3**

This release focuses on **numerical stability, deterministic behavior, ANN correctness, and distance kernel performance**.  
Several internal improvements were introduced to make SmartKNN more **robust, reproducible, and production-safe**.

A new **global structure distance parameter (`global_lambda`)** has also been introduced to incorporate dataset-level structure into neighbor ranking.

This release is **fully backward compatible**.

---

## New Feature

### Global Structure Distance (`global_lambda`)

A new parameter `global_lambda` allows SmartKNN to incorporate **global structural information** into the distance metric.

Distance computation now follows:

```
distance = local_distance + global_lambda * global_distance
```

Where:

- `local_distance` - standard weighted feature distance
- `global_distance` - dataset-level structural distance
- `global_lambda` - controls contribution of global structure


This ensures **full backward compatibility with previous versions** while enabling optional structural awareness.

In many datasets this improves neighbor quality and can yield **~1–3% accuracy improvements**.

---

# Core Improvements

## 1. Estimator Core (`SmartKNN`)

### Parameter Handling

- Strengthened `set_params()` validation
  - Parameters are validated against `get_params()`
  - Prevents silent addition of invalid attributes

### Input Validation

Added strict validation for:

- `k >= 1`
- `len(y) == X.shape[0]`
- `X` must be 2-D
- Feature count consistency during inference

### NaN / Inf Handling

Implemented robust `_replace_nonfinite()`:

- Handles:
  - `NaN`
  - `+Inf`
  - `-Inf`
- Columns with all non-finite values fall back to **0.0**

### Consistent Imputation

Training medians are now stored in:

```
self.impute_values_
```

Query data now uses **training medians**, ensuring:

- deterministic predictions
- query batch independence

### Feature Filtering Safety

Added protection against removing all features:

```
if X_f.shape[1] == 0:
raise ValueError("All features removed by weight_threshold")
```

### Classification Inference Improvements

Improved `_infer_classification()` heuristic.

Now detects:

- boolean labels
- low-cardinality integer labels
- string/object labels

Supported dtype kinds:

```
("U", "S", "O")
```

### Query Validation

Added checks in `_prepare_query()`:

- Single sample `(d,)` auto reshaped to `(1, d)`
- Feature mismatch detection
- Robust non-finite handling

### sklearn Compatibility

Added attribute:

```
self.n_features_in_
```

Improves compatibility with **scikit-learn style tooling**.

### Serialization Improvements

Improved pickling safety.

Backend objects are excluded during serialization.

Implemented:

- __getstate__()
- __setstate__()

Backends are automatically rebuilt during unpickling.

---

# 2. Neighbor Search Engine 

### Deterministic ANN Validation

ANN validation sampling is now deterministic:

```
rng = np.random.default_rng(0)
```

Ensures reproducible validation results.

### ANN Result Validation

Added safety checks for invalid ANN outputs:

- insufficient neighbors
- NaN / Inf distances

```
if not np.isfinite(dist_mat).all():
raise RuntimeError("ANN backend returned invalid distances")
```

### Candidate Neighbor Safety

Candidate size is now clamped:

```
cand_k = min(self.k * 5, n_samples)
```

Prevents backend instability.

### Distance Calculation Optimization

Reduced temporary allocations.

Old:

```
(diff ** 2) * weights
```

New:

```
diff * diff * weights
```

Improves performance and memory efficiency.

### Stable Neighbor Ordering

Top-k neighbors are now **sorted after `argpartition`**, ensuring deterministic outputs.

### Global Structure Integration

Distance computation now includes optional global structure contribution:

distance = local_distance + global_lambda * global_distance

This improves neighbor ranking without altering traditional KNN semantics.

---

# 3. Distance Kernel Layer 

### Safe Numeric Cleaning

Improved `_ensure_f32_clean()`:

- converts pandas inputs
- enforces contiguous `float32`
- safely replaces NaN / Inf

### Safer Infinite Handling

Previous behavior:

- posinf = 1e9
- neginf = -1e9

New behavior:

- posinf = 0.0
- neginf = 0.0

Prevents extreme distance explosions.

### Weight Validation

Added strict validation:

- correct dimensionality
- non-negative weights
- NaN/Inf protection
- minimum epsilon clamp

### Numba Optimized Kernels

Implemented high-performance kernels:

- `_weighted_l2_single`
- `_weighted_l2_batch`
- `_weighted_l2_multiquery`

Features:

- `parallel=True`
- cache-friendly blocking
- reduced temporary allocations

### Multi-Query Memory Guard

Added protection against accidental huge allocations:

```
projected_bytes = nq * nx * 4
```

---

# 4. ANN Backend

### ANN Import Safety

Exception handling refined:

Prevents masking real runtime errors.

### IVF Training Optimization

IVF now trains on a **random subset** instead of the full dataset:

```
train_size = min(200_000, n)
rng = np.random.default_rng(42)
train_idx = rng.choice(n, train_size, replace=False)
index.train(X[train_idx])
```

Improves training speed for very large datasets.

### Safe FAISS Neighbor Validation

Removed silent index correction.

Old behavior:

```
idx == -1 → replaced with 0
```

New behavior:

```
raise RuntimeError("ANN returned invalid neighbor index")
```


Prevents silent incorrect neighbors.

### nprobe Control

Added strict cap:

- nprobe ≤ 8

Ensures predictable ANN latency.

### Default nprobe Heuristic

Default probe count now approximates:

```
nprobe ≈ sqrt(nlist)
```

with maximum cap of **8**.

### GPU Resource Management

Stored GPU resource explicitly:

```
self.gpu_res = faiss.StandardGpuResources()
```

Prevents GPU lifetime issues.

### Query Memory Layout

Queries are now forced contiguous:

```
np.ascontiguousarray()
```

Ensures ANN compatibility.

### Large GPU Batch Handling

Large GPU batches now trigger **warnings instead of hard errors**.

---

# Logging Improvements

Library logging now follows application-friendly design.

```
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
```

Libraries no longer attach their own handlers.

---

# Numerical Stability Improvements

- Protected division by zero in distance weights
- Stable normalization using:

```
max(std, 1e-12)
```

- Removed unstable infinite replacements
- Consistent float32 pipelines

---

# Performance Improvements

- Reduced temporary arrays in distance computation
- Deterministic ANN validation sampling
- Cache-friendly Numba kernels
- Candidate neighbor clamping
- Faster ANN index training using subset sampling

---

# Summary

## Major Fixes

- Parameter handling bugs
- NaN / Inf robustness
- Serialization safety
- ANN backend correctness
- FAISS training strategy
- Neighbor search stability

## Major Improvements

- Faster distance kernels
- Deterministic ANN validation
- More stable inference
- Cleaner logging design
- Optional global structure distance

---

## Release 0.2.2 - **SmartKNN v2.2**
This release focuses on scalability, correctness of automatic decisions, and faster training on large-scale datasets, with multiple internal improvements across feature weighting and ANN indexing.

### Major Changes
- **Mutual Information (MI)** feature weighting optimized
  - **MI** computation is now **parallelized** for very high-dimensional datasets
  - Preserves exact behavior for low and medium dimensional data
- **Automatic backend** selection corrected
  - **Brute-force** backend is now explicitly enforced below **10K** rows
  - Prevents unnecessary ANN overhead on small datasets

- **Feature selection** improvements
  - **Random Forest–based** feature relevance updated with improved split constraints
  - More stable feature pruning under **noisy or skewed distributions**

- **ANN backend** training optimized for **very large datasets**
  - **Improved scalability** for **multi-million** row datasets
  - **Faster ANN index construction** without affecting **inference behavior**

### Performance Improvements
- Faster training time observed:
  - **~10% speedup on medium-sized** datasets
  - **~25% speedup on multi-million** row datasets
- **Reduced ANN index build overhead** for very large datasets
- **No regression in inference accuracy or latency**

### Bug Fixes
- **Fixed inference-time** Handling of **NAN / INF** values in Query Inputs.
  SmartKNN now consistently **emits a warning when invalid values are detected during prediction**, while preserving existing **normalization and prediction behavior**.


## Notes

- ANN inference behavior and tuning **(nlist, nprobe)** remain unchanged
- Improvements primarily affect **training-time scalability**
- No **API changes** 

---

## Release 0.2.1 - **SmartKNN v2.1**

This release focuses on **stability, proper dependencies, and runtime safety**, fixing scaling issues and ensuring seamless installation.

### Major Changes
- Fixed **scaling bugs** in large datasets and ultra-low latency inference
- Added **dependencies** to prevent runtime errors
- Full **classification and regression support**
- Optimized internal workflows for memory and speed

---

## Release 0.2.0 - **SmartKNN v2**

This release marks a **major milestone** for SmartKNN, introducing a fully optimized
and scalable architecture with high-performance **classification and regression**.

### Major Changes
- Full **classification support restored**
- **ANN backend** introduced for fast nearest-neighbor search
- **Brute-force backend** retained for small datasets
- Scales to **millions of rows** with low latency

### New Features
- **ANN backend** for fast prediction on large datasets  
  - Optional **GPU support** for neighbor search  
  - Tunable ANN parameters:
    - `nlist` — number of coarse clusters
    - `nprobe` — number of clusters searched per query
  - Safe default values provided
- **Automatic backend selection**
  - Brute backend for small datasets
  - ANN backend for medium and large datasets
- Full support for **classification and regression**
- Robust internal handling of **NaN / Inf values**
- **Distance-weighted voting** for classification
- **Feature masking** via weight-thresholding
- **Automatic evaluation utilities**
  - Unified `evaluate_auto` interface for classification and regression
  - Automatic task-type inference from target values
  - Built-in metrics:
    - Regression: MSE, RMSE, MAE, R²
    - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
  - Safe handling of NaN / Inf during evaluation
  - Supports non-numeric classification labels

### Performance Improvements
- Fully **vectorized NumPy** implementation
- **Numba acceleration** added for:
  - Distance computation
  - Core inner loops
- **Ultra-low latency inference** compared to v1
- Faster training compared to v1
- Stress-tested on large and heavy datasets

### Benchmarks
- Extensive benchmarking on:
  - Classification tasks
  - Regression tasks
  - Large-scale datasets
- Demonstrates:
  - Significant speedups over v1
  - Competitive CPU latency against tree-based models

### Known Limitations
- ANN quality depends on dataset characteristics and tuning
- GPU support is limited to neighbor search
- Probability calibration is not yet available

---

## Release 0.1.1 - **SmartKNN v1.1**

This release focuses on **stability and safety**.

### Changes
- **Classification disabled** due to correctness concerns
- Added explicit weight control parameters:
  - `alpha`
  - `beta`
  - `gamma`

### Notes
- Feature weight learning remains **accuracy-focused** and computationally expensive
- This release prioritizes **correctness over functionality**

---
## Release 0.1.0 - **SmartKNN v1 (Initial Release)**

**SmartKNN is born.**

This is the first public release introducing the core ideas behind SmartKNN.

### Core Features
- Automatic **feature weight learning**
  - Hybrid weighting using:
    - Univariate **MSE-based relevance**
    - **Mutual Information (MI)**
    - **Random Forest feature importance**
- Automatic **preprocessing**
- Automatic detection of:
  - Classification
  - Regression
- Internal handling of **NaN / Inf values**
- **Feature masking** via weight threshold
- Pure **Python + NumPy** implementation

### Design Characteristics
- Accuracy-first approach
- Not fully vectorized
- No ANN or GPU support
- Limited scalability for large datasets

### Known Issues
- Feature weight learning is **slow** and does not scale well
- Classification outputs returned **numeric values instead of labels**
- Focused on correctness rather than speed
