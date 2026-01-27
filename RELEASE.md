# SmartKNN Release Notes

---

## Release 0.2.2 — **SmartKNN v2.2**
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
