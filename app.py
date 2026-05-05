import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Failure Risk Dashboard",
    layout="centered"
)

st.title("Predictive Failure Risk Dashboard")
st.write(
    "A Gates-inspired ML demo that estimates failure risk from industrial operating conditions."
)

model = joblib.load("failure_model.pkl")

st.subheader("Operating Conditions")

air_temp = st.slider("Air temperature [K]", 295.0, 305.0, 300.0)
process_temp = st.slider("Process temperature [K]", 305.0, 315.0, 310.0)
rot_speed = st.slider("Rotational speed [rpm]", 1100, 3000, 1500)
torque = st.slider("Torque [Nm]", 3.0, 80.0, 40.0)
tool_wear = st.slider("Tool wear [min]", 0, 260, 100)

input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

risk = model.predict_proba(input_data)[0][1]

st.subheader("Prediction")
st.metric("Predicted Failure Risk", f"{risk:.2%}")

if risk < 0.20:
    st.success("Low risk: continue normal operation.")
elif risk < 0.50:
    st.warning("Medium risk: review operating conditions and monitor trend.")
else:
    st.error("High risk: flag for engineering review.")

st.subheader("Input Summary")
st.dataframe(input_data)

st.caption(
    "Demo concept: using historical operating data to support predictive maintenance, "
    "durability analysis, and engineering decision-making."
)