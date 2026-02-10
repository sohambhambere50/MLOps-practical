# predict.py
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input (sepal length, sepal width, petal length, petal width)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample_input)

print("🌸 Predicted class:", prediction[0])
