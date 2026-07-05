import math
import numpy as np
import streamlit as st


def estimate_risk(features):
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age = features[0]

    score = (
        0.03 * pregnancies
        + 0.025 * max(glucose - 100, 0) / 30
        + 0.015 * max(blood_pressure - 80, 0) / 20
        + 0.01 * max(skin_thickness - 20, 0) / 20
        + 0.001 * insulin / 100
        + 0.04 * max(bmi - 25, 0) / 10
        + 0.75 * dpf
        + 0.02 * max(age - 40, 0) / 20
        - 1.1
    )
    return 1 / (1 + math.exp(-score))


st.set_page_config(page_title="Diabetes Risk Checker", page_icon="🩺", layout="centered")
st.title("Diabetes Risk Checker")
st.caption("A simple and fast Streamlit interface for diabetes risk prediction.")

with st.form("diabetes_form"):
    st.subheader("Enter patient details")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        glucose = st.number_input("Glucose", min_value=50, max_value=250, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=140, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=5, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=20, max_value=900, value=100)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=20, max_value=90, value=40)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    features = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        dtype=float,
    )
    probability = estimate_risk(features)

    if probability >= 0.6:
        status = "High risk"
        color = "🔴"
    elif probability >= 0.4:
        status = "Moderate risk"
        color = "🟡"
    else:
        status = "Low risk"
        color = "🟢"

    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric("Risk Probability", f"{probability * 100:.1f}%")
    st.success(f"{color} {status}")
    st.info("This is a lightweight demo interface built for quick testing and presentation.")
