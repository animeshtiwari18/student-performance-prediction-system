import pandas as pd
import random
import os

os.makedirs("data", exist_ok=True)

rows = []

def generate_row(label):
    if label == "Good":
        return [
            random.randint(75, 95),
            random.randint(6, 9),
            random.randint(22, 30),
            random.randint(7, 10),
            round(random.uniform(7.8, 9.8), 1),
            "Good"
        ]
    elif label == "Average":
        return [
            random.randint(50, 70),
            random.randint(4, 6),
            random.randint(16, 20),
            random.randint(5, 6),
            round(random.uniform(6.0, 6.9), 1),
            "Average"
        ]
    else:
        return [
            random.randint(20, 45),
            random.randint(1, 3),
            random.randint(8, 14),
            random.randint(1, 4),
            round(random.uniform(3.0, 5.5), 1),
            "Poor"
        ]

# fixed required case
rows.append([52, 6, 18, 6, 6.5, "Average"])

for _ in range(700):
    rows.append(generate_row("Good"))

for _ in range(700):
    rows.append(generate_row("Average"))

for _ in range(599):
    rows.append(generate_row("Poor"))

columns = [
    "attendance",
    "study_hours",
    "internal_marks",
    "assignments",
    "previous_gpa",
    "performance"
]

df = pd.DataFrame(rows, columns=columns)
df.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully")
