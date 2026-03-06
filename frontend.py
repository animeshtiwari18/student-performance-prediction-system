import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="🎓 Student Dashboard", layout="wide")
st.title("🎓 Student Performance Dashboard")
st.write("---")

attendance = st.slider("Attendance (%)", 0, 100, 60)
study_hours = st.slider("Study Hours / day", 0, 10, 4)
internal_marks = st.slider("Internal Marks", 0, 30, 18)
assignments = st.slider("Assignments Submitted", 0, 10, 6)
previous_gpa = st.slider("Previous GPA", 0.0, 10.0, 6.5)
online_engagement = st.slider("Online Engagement (%)", 0, 100, 55)
late_submissions = st.slider("Late Submissions", 0, 10, 1)

payload = {
    "attendance": attendance,
    "study_hours": study_hours,
    "internal_marks": internal_marks,
    "assignments": assignments,
    "previous_gpa": previous_gpa,
    "online_engagement": online_engagement,
    "late_submissions": late_submissions
}

if st.button("🔮 Predict Performance"):

    result = requests.post("http://127.0.0.1:5000/predict", json=payload).json()

    st.header("Prediction Result")
    st.metric("Prediction", result["prediction"])
    st.metric("Risk Level", result["risk_level"])
    st.metric("Burnout Index", f'{result["burnout_index"]}%')
    st.metric("Model Accuracy", f'{result["confidence_score"]}%')

    # Probability Confidence
    st.header("📊 Prediction Confidence")
    st.write(result["confidence_text"])
    for grade, prob in result["probabilities"].items():
        st.write(f"{grade}: {prob}%")

    # Human AI Explanation
    st.header("🧠 AI Explanation")
    st.success(result["human_explanation"])

    # Grade Improvement
    st.header("📈 Grade Improvement Analysis")
    st.write("Next Target Grade:", result["next_grade_target"])
    st.write("Score Gap:", result["score_gap"])

    st.header("🤖 Intelligent AI Recommendation")
    st.success(result["upgrade_recommendation"])

    # Radar Chart
    st.header("📊 Academic Performance Radar")

    categories = [
        "Attendance",
        "Study Hours",
        "Internal Marks",
        "Assignments",
        "Previous GPA",
        "Online Engagement",
        "Late Submissions (Inverse)"
    ]

    values = [
        attendance,
        study_hours * 10,
        internal_marks * 3,
        assignments * 10,
        previous_gpa * 10,
        online_engagement,
        100 - (late_submissions * 10)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )

    st.plotly_chart(fig)

    # PDF Download
    pdf_res = requests.post("http://127.0.0.1:5000/generate_pdf", json=result)

    if pdf_res.status_code == 200:
        st.download_button(
            "Download Full AI Report (PDF)",
            data=pdf_res.content,
            file_name="Student_Performance_Report.pdf",
            mime="application/pdf"
        )