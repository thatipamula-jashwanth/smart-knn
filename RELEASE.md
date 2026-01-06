# SmartKNN Release Notes

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
