import numpy as np
from smart_knn import SmartKNN

np.random.seed(0)
X = np.random.rand(300, 3).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

model = SmartKNN(k=5)
model.fit(X, y)

preds = model.predict(X[:10])
print("Predictions:", preds)
print("Classes:", model.classes_)
