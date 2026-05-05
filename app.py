"""
app.py

Streamlit dashboard for predictive failure risk assessment.
Allows users to adjust industrial operating conditions and receive real-time failure risk predictions.

Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Predictive Failure Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔧 Predictive Failure Risk Dashboard")
st.markdown(
    """
    An AI-enabled predictive maintenance tool that estimates failure risk from industrial operating conditions.  
    **Use case**: Support engineering decision-making for product durability, testing, and field operations.
    """
)

# Load model and features
try:
    model = joblib.load("failure_model.pkl")
    features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error(
        "❌ Model files not found. Please run training first:\n"
        "```\n"
        "python train_model.py\n"
        "```"
    )
    st.stop()

# Sidebar for inputs
st.sidebar.header("Operating Conditions")
st.sidebar.markdown(
    "Adjust sliders to explore how operating conditions affect failure risk."
)

air_temp = st.sidebar.slider(
    "Air temperature [K]",
    min_value=295.0,
    max_value=305.0,
    value=300.0,
    step=0.5,
    help="Ambient air temperature in Kelvin"
)

process_temp = st.sidebar.slider(
    "Process temperature [K]",
    min_value=305.0,
    max_value=315.0,
    value=310.0,
    step=0.5,
    help="Process/component temperature in Kelvin"
)

rot_speed = st.sidebar.slider(
    "Rotational speed [rpm]",
    min_value=1100,
    max_value=3000,
    value=1500,
    step=50,
    help="Spindle/rotation speed in RPM"
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    min_value=3.0,
    max_value=80.0,
    value=40.0,
    step=1.0,
    help="Torque applied to tool/spindle in Newton-meters"
)

tool_wear = st.sidebar.slider(
    "Tool wear [min]",
    min_value=0,
    max_value=260,
    value=100,
    step=10,
    help="Cumulative tool wear in minutes"
)

# Prepare input data
input_data = pd.DataFrame([{
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": rot_speed,
    "Torque": torque,
    "Tool wear": tool_wear
}])

# Make prediction
risk_probability = model.predict_proba(input_data)[0][1]
predicted_class = model.predict(input_data)[0]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Failure Risk Prediction")
    
    # Risk metric with color coding
    if risk_probability < 0.20:
        risk_color = "green"
        risk_level = "🟢 LOW RISK"
        recommendation = "✓ Continue normal operation. Monitor tool wear periodically."
    elif risk_probability < 0.50:
        risk_color = "orange"
        risk_level = "🟡 MEDIUM RISK"
        recommendation = "⚠ Review operating conditions. Consider preventive maintenance schedule."
    else:
        risk_color = "red"
        risk_level = "🔴 HIGH RISK"
        recommendation = "🛑 Flag for engineering review. Plan maintenance intervention."
    
    # Display risk percentage
    st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{risk_probability:.1%}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>{risk_level}</h4>", unsafe_allow_html=True)
    
    # Recommendation
    st.info(recommendation)

with col2:
    st.subheader("Risk Category")
    st.markdown(
        f"""
        **Predicted Class:** {'Failure' if predicted_class == 1 else 'No Failure'}
        
        **Confidence:** {max(model.predict_proba(input_data)[0]) * 100:.1f}%
        """
    )

# Input summary
st.subheader("Current Operating Conditions")
st.dataframe(
    input_data.T.rename(columns={0: "Value"}),
    use_container_width=True
)

# Feature importance
st.subheader("Feature Importance")
st.markdown("Impact of each operating condition on failure risk (from model training).")

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance for Failure Risk Prediction')
ax.invert_yaxis()
plt.tight_layout()
st.pyplot(fig)

# Footer
st.divider()
st.markdown(
    """
    ### About This Tool
    
    **Purpose**: Demonstrate predictive maintenance and failure risk assessment workflows.  
    **Model**: RandomForest classifier trained on UCI AI4I 2020 Predictive Maintenance Dataset.  
    **Dataset**: 10,000 synthetic operating records from an industrial machining process.  
    **Use**: This is a proof-of-concept; production models would include real asset/field data, 
    advanced explainability (SHAP), and continuous validation pipelines.
    
    **Learn More**: See [README](https://github.com/rothluk00044/Predictive-Failure-Risk-Dashboard) for setup, architecture, and next steps.
    """
)