# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('model/student_model.pkl')
encoders = joblib.load('model/encoders.pkl')
target_encoder = joblib.load('model/target_encoder.pkl')

st.title("🎓 EduStat – Student Risk Prediction")
st.write("Offline ML app to detect at-risk students in Nigeria.")

# Input form
with st.form("student_form"):
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", 10, 20)
    class_level = st.selectbox("Class Level", ['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'])
    attendance = st.slider("Attendance Rate (%)", 30.0, 100.0, 85.0)
    math = st.slider("Math Score", 0.0, 100.0, 60.0)
    english = st.slider("English Score", 0.0, 100.0, 62.0)
    science = st.slider("Science Score", 0.0, 100.0, 58.0)
    study_hours = st.slider("Study Hours/Week", 0, 24, 10)
    parental_support = st.selectbox("Parental Support", ['Yes', 'No'])
    food_security = st.selectbox("Regular Meals Before School?", ['Yes', 'No'])
    submitted = st.form_submit_button("Predict Risk")

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

    # Encode input
    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predict
    prediction = model.predict(input_df)[0]
    risk_label = target_encoder.inverse_transform([prediction])[0]

    if risk_label == "Yes":
        st.error("⚠️ The student is AT RISK. Please take action.")
    else:
        st.success("✅ The student is NOT at risk. Keep supporting their learning!")

