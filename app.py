from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
import shap

app = Flask(__name__)
CORS(app)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("student_dataset.csv")

X = df.drop("performance", axis=1)
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    random_state=42
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- HELPERS ----------------
def calculate_burnout(data):
    return int((data["late_submissions"] * 5) +
               (100 - data["attendance"]) * 0.2)

def calculate_score(data):
    return (
        data["attendance"] * 0.2 +
        data["study_hours"] * 5 +
        data["internal_marks"] * 1.5 +
        data["assignments"] * 2 +
        data["previous_gpa"] * 5 +
        data["online_engagement"] * 0.1 -
        data["late_submissions"] * 3
    )

# ---------------- NEW OVERALL RISK SCORE ----------------
def calculate_overall_risk_score(burnout, score, prediction):

    if score >= 130:
        score_risk = 0
    else:
        score_risk = (130 - score)

    if prediction == "Low":
        prediction_risk = 70
    elif prediction == "Average":
        prediction_risk = 40
    else:
        prediction_risk = 10

    overall_risk = (
        0.4 * burnout +
        0.4 * score_risk +
        0.2 * prediction_risk
    )

    return round(min(overall_risk, 100), 2)

# ---------------- RISK ENGINE ----------------
def calculate_risk_level(burnout):
    if burnout > 70:
        return "High Risk"
    elif burnout > 40:
        return "Moderate Risk"
    else:
        return "Low Risk"

# ---------------- UPGRADE ENGINE ----------------
def intelligent_upgrade(data):

    score = calculate_score(data)

    if score >= 130:
        return (
            "Student is already performing at High level. "
            "Maintain consistency in study hours, internal marks, assignments, GPA and attendance."
        )

    if score < 100:
        target_grade = "Average"
        target_score = 100
    else:
        target_grade = "High"
        target_score = 130

    gap = round(target_score - score, 2)
    explanation = []
    portion = gap / 6

    explanation.append(
        f"- Increase study hours by {round(min((portion/5),10-data['study_hours']),2)} hours/day"
    )
    explanation.append(
        f"- Improve GPA by {round(min((portion/5),10-data['previous_gpa']),2)} points"
    )
    explanation.append(
        f"- Improve internal marks by {round(min((portion/1.5),30-data['internal_marks']),2)} marks"
    )
    explanation.append(
        f"- Submit {round(min((portion/2),10-data['assignments']),2)} more assignments"
    )
    explanation.append(
        f"- Improve attendance by {round(min((portion/0.2),100-data['attendance']),2)}%"
    )
    explanation.append(
        f"- Increase online engagement by {round(min((portion/0.1),100-data['online_engagement']),2)}%"
    )

    if data["late_submissions"] > 0:
        explanation.append(
            f"- Reduce late submissions by up to {data['late_submissions']}"
        )

    return (
        f"To upgrade to {target_grade}, approximately {gap} additional score points are required. "
        f"A balanced improvement plan:\n" +
        "\n".join(explanation)
    )

# ---------------- SHAP EXPLANATION ----------------
def generate_human_explanation(data):
    X_input = pd.DataFrame([data])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    predicted_class = model.predict(X_input)[0]
    predicted_index = model.predict_proba(X_input).argmax(axis=1)[0]

    values = shap_values.values[0][:, predicted_index]
    explanation = dict(zip(X.columns, values.tolist()))

    sorted_explanation = dict(
        sorted(explanation.items(),
               key=lambda x: abs(x[1]),
               reverse=True)
    )

    key_factors = [
        f.replace("_", " ").title()
        for f in list(sorted_explanation.keys())[:4]
    ]

    if predicted_class == "Low":
        return (
            f"The system predicts Low academic performance. "
            f"The primary contributing academic weaknesses include {', '.join(key_factors)}."
        )
    elif predicted_class == "Average":
        return (
            f"The student demonstrates moderate academic performance. "
            f"Key influencing factors include {', '.join(key_factors)}."
        )
    else:
        return (
            f"The student shows strong academic indicators. "
            f"The performance is mainly driven by {', '.join(key_factors)}."
        )

# ---------------- MAIN PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X_input = pd.DataFrame([data])

    ml_prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    class_labels = model.classes_

    prob_dict = {
        class_labels[i]: round(float(probabilities[i] * 100), 2)
        for i in range(len(class_labels))
    }

    confidence = max(prob_dict.values())

    confidence_text = (
        f"The model is approximately {confidence}% confident "
        f"that the student belongs to the '{ml_prediction}' category."
    )

    score = calculate_score(data)

    # Final grade based on score only
    if score < 100:
        prediction = "Low"
    elif score < 130:
        prediction = "Average"
    else:
        prediction = "High"

    burnout = calculate_burnout(data)
    risk = calculate_risk_level(burnout)
    overall_risk_score = calculate_overall_risk_score(burnout, score, prediction)
    human_explanation = generate_human_explanation(data)

    if prediction == "Low":
        next_grade = "Average"
        gap = 100 - score
    elif prediction == "Average":
        next_grade = "High"
        gap = 130 - score
    else:
        next_grade = "Highest Achieved"
        gap = 0

    upgrade_advice = intelligent_upgrade(data)

    return jsonify({
        "prediction": prediction,
        "probabilities": prob_dict,
        "confidence_text": confidence_text,
        "risk_level": risk,
        "burnout_index": burnout,
        "overall_risk_score": overall_risk_score,
        "confidence_score": round(accuracy * 100, 2),
        "human_explanation": human_explanation,
        "next_grade_target": next_grade,
        "score_gap": round(gap, 2),
        "upgrade_recommendation": upgrade_advice
    })

# ---------------- PDF ----------------
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.get_json()

    file_path = "Student_Performance_Report.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []
    style = ParagraphStyle(name='Normal', fontSize=12)

    elements.append(Paragraph("Student Performance Report", style))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Prediction: {data['prediction']}", style))
    elements.append(Paragraph(f"Risk Level: {data['risk_level']}", style))
    elements.append(Paragraph(f"Burnout Index: {data['burnout_index']}%", style))
    elements.append(Paragraph(f"Overall Risk Score: {data['overall_risk_score']}", style))
    elements.append(Paragraph(data["confidence_text"], style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(data["human_explanation"], style))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Grade Improvement Analysis:", style))
    elements.append(Paragraph(f"Next Target Grade: {data['next_grade_target']}", style))
    elements.append(Paragraph(f"Score Gap: {data['score_gap']}", style))
    elements.append(Paragraph(data["upgrade_recommendation"], style))

    doc.build(elements)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)