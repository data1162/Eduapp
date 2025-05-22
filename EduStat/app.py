import os
import joblib
import streamlit as st
import pandas as pd

# === Load Model and Encoders ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model', 'student_model.pkl'))
encoders = joblib.load(os.path.join(BASE_DIR, 'model', 'encoders.pkl'))
target_encoder = joblib.load(os.path.join(BASE_DIR, 'model', 'target_encoder.pkl'))

# === Streamlit App ===
st.set_page_config(page_title="EduStat - Student Risk Prediction", layout="centered")

st.title("🎓 EduStat – Predict At-Risk Nigerian Students")
st.write("""
Welcome to EduStat — a lightweight, offline-first machine learning app that helps teachers
and school administrators predict students at risk of poor performance in Nigerian schools.
""")

# === Form ===
with st.form("student_form"):
    st.subheader("Enter Student Details")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", 10, 20)
    class_level = st.selectbox("Class Level", ['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'])
    attendance = st.slider("Attendance Rate (%)", 30.0, 100.0, 85.0)
    math = st.slider("Math Score", 0.0, 100.0, 60.0)
    english = st.slider("English Score", 0.0, 100.0, 62.0)
    science = st.slider("Science Score", 0.0, 100.0, 58.0)
    study_hours = st.slider("Study Hours per Week", 0, 24, 10)
    parental_support = st.selectbox("Parental Support", ['Yes', 'No'])
    food_security = st.selectbox("Regular Meals Before School?", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict Risk")

# === Prediction ===
if submitted:
    input_df = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'class_level': class_level,
        'attendance_rate': attendance,
        'math_score': math,
        'english_score': english,
        'science_score': science,
        'study_hours': study_hours,
        'parental_support': parental_support,
        'food_security': food_security
    }])

    # Encode input using loaded encoders
    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    risk_label = target_encoder.inverse_transform([prediction])[0]

    st.subheader("📊 Prediction Result")
    if risk_label == "Yes":
        st.error("⚠️ The student is AT RISK of poor performance. Consider intervention.")
    else:
        st.success("✅ The student is NOT at risk. Continue current support strategies.")

    st.markdown("---")
    st.caption("EduStat powered by machine learning. Built by Ibrahim Ali © 2025.")
