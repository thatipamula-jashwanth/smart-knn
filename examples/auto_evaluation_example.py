import numpy as np
from smart_knn import SmartKNN
from smart_knn.evaluation import evaluate_auto

np.random.seed(0)
X = np.random.rand(200, 3).astype(np.float32)
y = (X[:, 0] > 0.5).astype(int)

model = SmartKNN(k=5)
model.fit(X, y)

preds = model.predict(X)
metrics = evaluate_auto(y, preds)

print(metrics)
