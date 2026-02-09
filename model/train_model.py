import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# ----------------- Generate Synthetic Dataset -----------------
np.random.seed(42)
n = 800

attendance = np.random.randint(40, 100, n)
study_hours = np.random.randint(0, 10, n)
internal_marks = np.random.randint(5, 30, n)
assignments = np.random.randint(0, 10, n)
previous_gpa = np.round(np.random.uniform(4.0, 9.5, n), 2)
online_engagement = np.random.randint(20, 100, n)
late_submissions = np.random.randint(0, 5, n)

X = np.column_stack([
    attendance,
    study_hours,
    internal_marks,
    assignments,
    previous_gpa,
    online_engagement,
    late_submissions
])

# ----------------- Target Variable -----------------
y = []
for i in range(n):
    if attendance[i] > 75 and study_hours[i] > 4 and previous_gpa[i] > 7:
        y.append("Good")
    elif attendance[i] > 55:
        y.append("Average")
    else:
        y.append("Poor")

y = np.array(y)

# ----------------- Encode Labels -----------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ----------------- Train Model -----------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y_encoded)

# ----------------- Save Model -----------------
joblib.dump(model, "student_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model trained and saved successfully")
