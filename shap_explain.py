import shap
import pickle
import numpy as np

# Load trained pipeline
with open("model/saved_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Extract parts from pipeline
scaler = pipeline.named_steps["scaler"]
model = pipeline.named_steps["model"]

# SHAP explainer for RandomForest
explainer = shap.TreeExplainer(model)

feature_names = [
    "Attendance",
    "Study Hours",
    "Internal Marks",
    "Assignments",
    "Previous GPA"
]

def explain_prediction(features):
    X = np.array(features).reshape(1, -1)

    # Apply same scaling as training
    X_scaled = scaler.transform(X)

    shap_values = explainer.shap_values(X_scaled)

    explanation = {}
    for i, name in enumerate(feature_names):
        explanation[name] = round(float(shap_values[1][0][i]), 4)

    return explanation
