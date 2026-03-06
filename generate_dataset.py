import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1200
data = []

for _ in range(rows):
    attendance = np.random.randint(40, 101)
    study_hours = np.random.randint(1, 11)
    internal_marks = np.random.randint(5, 31)
    assignments = np.random.randint(2, 11)
    previous_gpa = round(np.random.uniform(4.0, 10.0), 2)
    online_engagement = np.random.randint(30, 101)
    late_submissions = np.random.randint(0, 6)

    score = (
        attendance * 0.2 +
        study_hours * 5 +
        internal_marks * 1.5 +
        assignments * 2 +
        previous_gpa * 5 +
        online_engagement * 0.1 -
        late_submissions * 3
    )

    if score > 160:
        performance = "High"
    elif score > 120:
        performance = "Average"
    else:
        performance = "Low"

    data.append([
        attendance, study_hours, internal_marks,
        assignments, previous_gpa,
        online_engagement, late_submissions,
        performance
    ])

df = pd.DataFrame(data, columns=[
    "attendance", "study_hours", "internal_marks",
    "assignments", "previous_gpa",
    "online_engagement", "late_submissions",
    "performance"
])

df.to_csv("student_dataset.csv", index=False)
print("Dataset Generated Successfully ✅")