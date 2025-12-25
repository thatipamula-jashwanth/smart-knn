import numpy as np
import joblib
from smart_knn import SmartKNN

X = np.random.rand(500, 4).astype(np.float32)
y = np.random.randn(500)

model = SmartKNN(k=5)
model.fit(X, y)

joblib.dump(model, "smartknn_model.joblib")

loaded = joblib.load("smartknn_model.joblib")
preds = loaded.predict(X[:3])

print("Loaded model predictions:", preds)
