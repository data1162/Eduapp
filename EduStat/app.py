import os
import joblib
import streamlit as st
import pandas as pd

# === Load Model and Encoders ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model', 'student_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'model', 'encoders.pkl'))
target_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'target_encoder.pkl'))

# === App Config ===
st.set_page_config(page_title="EduStat - Smart School Assistant", layout="centered")

# === Title ===
st.markdown("""
# 🎓 EduStat – Smart Risk Predictor for Nigerian Students
Identify struggling students, analyze academic trends, and improve performance — all offline.
""", unsafe_allow_html=True)
st.divider()

# === Input Form ===
with st.form("student_form"):
    st.subheader("📥 Student Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.slider("Age", 10, 20)
        class_level = st.selectbox("Class Level", ['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'])
        parental_support = st.selectbox("Parental Support", ['Yes', 'No'])
        food_security = st.selectbox("Regular Meals Before School?", ['Yes', 'No'])
        study_hours = st.slider("Weekly Study Hours", 0, 24, 10)

    with col2:
        attendance = st.slider("Attendance Rate (%)", 30.0, 100.0, 85.0)
        math = st.slider("Current Math Score", 0.0, 100.0, 60.0)
        prev_math = st.slider("Previous Math Score", 0.0, 100.0, 55.0)
        english = st.slider("Current English Score", 0.0, 100.0, 62.0)
        prev_english = st.slider("Previous English Score", 0.0, 100.0, 60.0)
        science = st.slider("Current Science Score", 0.0, 100.0, 58.0)
        prev_science = st.slider("Previous Science Score", 0.0, 100.0, 56.0)

    submitted = st.form_submit_button("🔍 Predict Student Risk")

# === Process and Predict ===
if submitted:
    input_df = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'class_level': class_level,
        'attendance_rate': attendance,
        'math_score': math,
        'english_score': english,
        'science_score': science,
        'prev_math_score': prev_math,
        'prev_english_score': prev_english,
        'prev_science_score': prev_science,
        'study_hours': study_hours,
        'parental_support': parental_support,
        'food_security': food_security
    }])

    # Encode categorical inputs
    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    risk_label = target_encoder.inverse_transform([prediction])[0]

    # Weakest subject
    subject_scores = {
        'Math': math,
        'English': english,
        'Science': science
    }
    weakest_subject = min(subject_scores, key=subject_scores.get)

    # Trend Detection
    trend_math = "↑ Improving" if math > prev_math else "↓ Declining"
    trend_eng = "↑ Improving" if english > prev_english else "↓ Declining"
    trend_sci = "↑ Improving" if science > prev_science else "↓ Declining"

    # Readiness Score
    avg_score = (math + english + science) / 3
    readiness_score = round((attendance * 0.4 + avg_score * 0.6), 2)

    # === Display Results ===
    st.markdown("### 📊 Prediction Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        if risk_label == "Yes":
            st.error("⚠️ Student is AT RISK")
        else:
            st.success("✅ Student is NOT at Risk")

    with col2:
        st.metric("📘 Weakest Subject", weakest_subject)

    with col3:
        st.metric("📈 Readiness Score", f"{readiness_score}/100")

    st.divider()
    st.markdown("### 📉 Subject Trends")
    st.info(f"📐 **Math:** {trend_math} | 📖 **English:** {trend_eng} | 🔬 **Science:** {trend_sci}")

    st.markdown("✅ Use this information to prioritize targeted intervention or support sessions.")

    st.caption("EduStat | Built by Ibrahim Ali © 2025 • Offline ML for Nigerian schools.")
