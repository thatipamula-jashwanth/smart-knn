import numpy as np
from smart_knn import SmartKNN

np.random.seed(42)
X = np.random.rand(10_000, 8).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

model = SmartKNN(
    k=5,
    backend="ann",
    ann_quality_check=True
)

model.fit(X, y)

q = np.random.rand(5, 8).astype(np.float32)
preds = model.predict(q)

print("Fast ANN predictions:", preds)
