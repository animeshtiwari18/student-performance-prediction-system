import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎓 Student Dashboard", layout="wide")

# ---------- TITLE ----------
st.markdown(
    "<h1 style='text-align:center;color:navy;'>🎓 Student Performance Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("---")

# ================= INPUT =================
st.header("1️⃣ Student Performance Overview")

attendance = st.slider("📌 Attendance (%)", 0, 100, 60)
study_hours = st.slider("📚 Study Hours/day", 0, 10, 4)
internal_marks = st.slider("📝 Internal Marks", 0, 30, 18)
assignments = st.slider("📄 Assignments Submitted", 0, 10, 6)
previous_gpa = st.slider("🎓 Previous GPA", 0.0, 10.0, 6.5)
online_engagement = st.slider("💻 Online Engagement (%)", 0, 100, 55)
late_submissions = st.slider("⏰ Late Submissions", 0, 10, 1)

st.write("---")

# ================= PREDICTION =================
if st.button("🔮 Predict Performance"):
    payload = {
        "attendance": attendance,
        "study_hours": study_hours,
        "internal_marks": internal_marks,
        "assignments": assignments,
        "previous_gpa": previous_gpa,
        "online_engagement": online_engagement,
        "late_submissions": late_submissions
    }

    try:
        res = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = res.json()

        # ---------- SUMMARY ----------
        st.header("2️⃣ Performance Analysis (Input Based)")

        st.metric("Prediction", result["prediction"])
        st.metric("Risk Level", result["risk_level"])
        st.metric("Confidence Score", f"{result['confidence_score']}%")

        st.info(
            f"This student is at **{result['risk_level']}** "
            f"with a burnout index of **{result['burnout_index']}%**."
        )

        # ================= INPUT-BASED GRAPH =================
        st.subheader("📊 Student Input Performance (Easy to Understand)")

        labels = [
            "Attendance",
            "Study Hours",
            "Internal Marks",
            "Assignments",
            "Previous GPA",
            "Online Engagement",
            "Late Submissions"
        ]

        # 🔁 DIRECT mapping from inputs (NO hidden logic)
        values = [
            attendance,                         # 0–100
            (study_hours / 10) * 100,           # 0–10 → %
            (internal_marks / 30) * 100,        # 0–30 → %
            (assignments / 10) * 100,           # 0–10 → %
            (previous_gpa / 10) * 100,          # 0–10 → %
            online_engagement,                  # 0–100
            100 - (late_submissions / 10) * 100 # more late = worse
        ]

        # Traffic-light colors
        colors = []
        for v in values:
            if v >= 70:
                colors.append("green")
            elif v >= 40:
                colors.append("orange")
            else:
                colors.append("red")

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, values, color=colors)

        ax.axhline(70, linestyle="--")
        ax.axhline(40, linestyle="--")

        ax.set_ylim(0, 100)
        ax.set_ylabel("Performance (%)")
        ax.set_title("🟢 Good | 🟡 Average | 🔴 Needs Improvement")

        plt.xticks(rotation=30)

        # Show values on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 2,
                f"{int(h)}%",
                ha="center",
                fontsize=9
            )

        st.pyplot(fig)

        plt.clf()
        plt.close()

        st.info(
            "This graph directly reflects student inputs. "
            "Any change in sliders immediately updates the graph."
        )

        # ---------- SUGGESTIONS ----------
        st.header("3️⃣ Personalized Support & Suggestions")

        st.success(result["suggested_action"])

        with st.expander("🔍 Key Weak Areas"):
            for r in result["key_reasons"]:
                st.write(f"❌ {r}")

        with st.expander("💡 Recommended Interventions"):
            for i in result["interventions"]:
                st.write(f"✅ {i}")

        st.caption(
            "⚖️ This system is for academic support only. "
            "Predictions are advisory and privacy-focused."
        )

        st.balloons()

    except Exception as e:
        st.error(f"Backend not reachable: {e}")
