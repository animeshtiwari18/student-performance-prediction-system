import os
from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ======================================================
# LOAD TRAINED ML MODEL (from /model folder)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "student_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ======================================================
# PREDICTION API
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[ 
        float(data["attendance"]),
        float(data["study_hours"]),
        float(data["internal_marks"]),
        float(data["assignments"]),
        float(data["previous_gpa"]),
        float(data["online_engagement"]),
        float(data["late_submissions"])
    ]])

    # ---------- ML Prediction ----------
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # ---------- Confidence ----------
    probabilities = model.predict_proba(features)[0]
    confidence = round(np.max(probabilities) * 100, 2)
    burnout_index = round(100 - confidence, 2)

    if confidence >= 70:
        risk_level = "Low Risk"
    elif confidence >= 40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    # ---------- Explainability (Rule-based for clarity) ----------
    key_factors = {
        "Attendance": data["attendance"]/100 - 0.5,
        "Study Hours": data["study_hours"]/5 - 0.5,
        "Internal Marks": data["internal_marks"]/15 - 0.5,
        "Assignments": data["assignments"]/5 - 0.5,
        "Previous GPA": data["previous_gpa"]/6 - 0.5,
        "Online Engagement": data["online_engagement"]/50 - 0.5,
        "Late Submissions": -data["late_submissions"]/5 + 0.5
    }

    negative_factors = [k for k, v in key_factors.items() if v < 0]

    # ---------- Interventions ----------
    interventions = []
    if data["attendance"] < 60:
        interventions.append("📢 Improve class attendance")
    if data["study_hours"] < 3:
        interventions.append("📖 Increase daily study hours")
    if data["previous_gpa"] < 6:
        interventions.append("🎓 Academic counselling suggested")
    if data["late_submissions"] > 0:
        interventions.append("📝 Reduce late submissions")
    if data["online_engagement"] < 50:
        interventions.append("💻 Participate more in online learning")

    if not interventions:
        interventions.append("✅ Maintain current performance")

    interventions = interventions[:3]

    suggested_action = (
        f"Focus on improving {', '.join(negative_factors[:2])}"
        if negative_factors else
        "Maintain consistency in academic performance"
    )

    return jsonify({
        "prediction": predicted_label,
        "risk_level": risk_level,
        "confidence_score": confidence,
        "burnout_index": burnout_index,
        "key_factors": key_factors,
        "key_reasons": negative_factors,
        "interventions": interventions,
        "suggested_action": suggested_action
    })


# ======================================================
# RUN SERVER
# ======================================================
if __name__ == "__main__":
    app.run(debug=True)
