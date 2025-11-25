# SmartKNN Design Overview

This document explains the **internal architecture** and **design decisions** behind SmartKNN.
It is intended for contributors, advanced users, and anyone who wants to understand how the algorithm works internally.

---

#  High-Level Architecture

SmartKNN improves traditional KNN by adding a full preprocessing and feature-weighting pipeline before performing nearest-neighbor search.

The core processing pipeline is:

```
Raw Input Data -->> Sanitization -->> Normalization -->> Feature Weight Learning
               -->> Feature Filtering -->> Weighted Distance Search -->> Prediction
```

---

#  Core Components

SmartKNN consists of several modular components, each responsible for a distinct function.

## 1. `weight_learning.py`

Learns feature weights using three scoring strategies:

### **1. Univariate MSE Weighting**

* Fit a linear regression for each feature independently
* Compute MSE between predictions and actual values
* Weight = 1 / (MSE + eps)

### **2. Mutual Information Weighting**

* Computes dependency between each feature and the target
* Useful for nonlinear relationships

### **3. Random Forest Feature Importance**

* Uses ensemble tree importance
* Captures complex feature interactions

### **Final Weight**

Weights are combined:

```python
alpha * mse + beta * mi + gamma * rf
```

and then normalized.

---

## 2. `data_processing.py`

Handles **input cleaning and feature filtering**.

### **NaN/Inf Sanitization**

* Replaces NaN with column medians
* Replaces Inf with ±1e9

### **Feature Filtering**

Removes features whose weights are below the threshold.
Ensures at least `min_features` survive.

---

## 3. `utils.py`

Utility functions for:

### **Normalization**

* Computes mean and std
* Applies standard scaling

### **Type Enforcement**

* Ensures numpy arrays
* Ensures numeric dtypes

---

## 4. `distance.py`

Implements optimized **weighted Euclidean distance**.

### **Weighted Euclidean**

For vectors `a` and `b`:

```
D(a, b) = sqrt( Σ w[i] * (a[i] - b[i])² )
```

### **Batch Distance**

Uses fast vectorized NumPy operations:

```
(X - query)² * weights
```

### **Multiquery Distance**

Efficiently computes distance from multiple queries to multiple samples.

---

#  Putting It All Together — The SmartKNN Class

## `fit(X, y)` Pipeline

1. Validate schema (numeric only)
2. Convert to numpy
3. Sanitize NaN / Inf
4. Normalize features using mean/std
5. Learn feature weights
6. Filter low-weight features
7. Store:

   * filtered X
   * filtered weights
   * feature mask
   * normalization parameters

## `predict(Xq)` Pipeline

1. Sanitize query NaN/Inf
2. Normalize query
3. Apply feature mask
4. Compute weighted distances
5. Take k nearest neighbors
6. Output mean (regression) or mode (classification)

---

#  Performance Considerations

SmartKNN is **not yet optimized for speed** — it currently focuses on **accuracy, stability, and feature intelligence**, not runtime.

### Current State (Honest Overview)

* Weighted distance computation is **vectorized but still slower** than classic KNN
* Feature-weight learning uses **MSE, MI, and Random Forest**, which adds overhead
* Prediction time can be slower because:

  * More preprocessing steps
  * Weighted Euclidean adds extra multiplications
  * No prototype compression yet
  * No clustering acceleration yet

### Why SmartKNN Is Slower Right Now

SmartKNN performs **extra intelligent steps** that sklearn KNN does NOT:

* Cleaning + sanitizing data
* Normalizing
* Learning weights
* Filtering weak features
* Applying weighted distance

This adds computational cost.

### Planned Performance Improvements

* Prototype compression (store fewer representative points)
* Multi-stage clustering (cluster -->> KNN inside cluster)
* Distance signature caching
* GPU acceleration for batch distances
* NumPy/Numba/CuPy JIT optimization
* Approximate nearest neighbor support (FAISS)

These upgrades will make SmartKNN significantly faster in future versions.

---

#  Extensibility

The system is designed to support future extensions:

### Planned Modules

* `adaptive_k.py`: Auto-tuning of K per sample
* `prototypes.py`: Reduce dataset via representative points
* `signatures.py`: Cache reusable distance patterns

Each piece is modular, making the system easy to modify and experiment with.

---

#  Design Goals

* **Accuracy**: Better than traditional KNN through weighting
* **Robustness**: Handle NaN, Inf, scaling, noise automatically
* **Ease of Use**: scikit-learn-like API
* **Speed**: Efficient distance computations
* **Modularity**: Easy for contributors to extend

---

#  Diagram of Data Flow

```
                 ┌──────────────────────┐
                       Raw Data        
                 └──────────┬───────────┘
                            │
             ┌──────────────▼──────────────┐
                Cleaning & Sanitization     
             └──────────────┬──────────────┘
                            │
                     ┌-──────▼──────┐
                       Normalization
                     └──────┬───────┘
                            │
                ┌───────────▼───────────┐
                    Feature Weighting    
                └───────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                    Feature Filtering   
                 └──────────┬───────────┘
                            │
                 ┌──────────▼───────────┐
                   Weighted Distance KNN 
                 └──────────┬───────────┘
                            │
                      ┌─────▼────-─┐
                        Prediction 
                      └────────────┘
```

---

#  Contributing

If you want to add new modules (adaptive_k, prototypes, etc.), follow the modular design shown here.

PRs are welcome!

---
