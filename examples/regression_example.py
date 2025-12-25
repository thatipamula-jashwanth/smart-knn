import numpy as np
from smart_knn import SmartKNN

np.random.seed(42)
X = np.random.rand(200, 4).astype(np.float32)
y = 3 * X[:, 0] + 2 * X[:, 1] + 0.1 * np.random.randn(200)


model = SmartKNN(k=5)
model.fit(X, y)

preds = model.predict(X[:5])

print("Predictions:", preds)
