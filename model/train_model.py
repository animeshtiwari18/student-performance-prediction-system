import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "student_dataset.csv")
df = pd.read_csv(dataset_path)

X = df.drop("performance", axis=1)
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save both model and accuracy
model_path = os.path.join(BASE_DIR, "model", "student_model.pkl")
joblib.dump({
    "model": model,
    "accuracy": accuracy
}, model_path)

print(f"Model saved with accuracy: {round(accuracy*100,2)}% ✅")