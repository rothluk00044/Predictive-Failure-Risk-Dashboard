"""
app.py

Streamlit dashboard for predictive failure risk assessment.
Demonstrates applied ML for engineering decision support in industrial settings.

Run with: python -m streamlit run app.py

Features:
- Real-time risk assessment with dynamic sliders
- Sensitivity analysis: how one variable affects risk while others stay fixed
- Risk trend simulation: illustrative demo of risk over time
- Improved risk driver explanation: engineering-focused interpretation
- Scenario comparison: current vs. conservative baseline
- Feature importance and model performance metrics
- Engineering interpretation and failure mode guidance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sensitivity_analysis(model, features, base_input, variable_to_analyze):
    """
    Generate sensitivity analysis for a single variable.
    
    Args:
        model: Trained ML model
        features: List of feature names
        base_input: DataFrame with baseline values
        variable_to_analyze: Feature name to vary
    
    Returns:
        DataFrame with varied values and predictions
    """
    # Define ranges for each variable
    ranges = {
        "Air temperature": np.linspace(295, 305, 25),
        "Process temperature": np.linspace(305, 315, 25),
        "Rotational speed": np.linspace(1100, 3000, 25),
        "Torque": np.linspace(3, 80, 25),
        "Tool wear": np.linspace(0, 260, 25)
    }
    
    varied_values = ranges[variable_to_analyze]
    results = []
    
    for value in varied_values:
        input_copy = base_input.copy()
        input_copy[variable_to_analyze] = value
        risk_prob = model.predict_proba(input_copy)[0][1]
        results.append({
            "variable_value": value,
            "failure_risk": risk_prob * 100
        })
    
    return pd.DataFrame(results)

def generate_risk_trend_demo(model, features, base_input, num_points=25):
    """
    Generate illustrative risk trend over time (synthetic demo).
    
    Tool wear increases gradually, other values vary slightly.
    """
    results = []
    
    initial_tool_wear = base_input["Tool wear"].values[0]
    
    for i in range(num_points):
        time_fraction = i / (num_points - 1) if num_points > 1 else 0
        
        input_copy = base_input.copy()
        
        # Gradually increase tool wear
        input_copy["Tool wear"] = initial_tool_wear + (260 - initial_tool_wear) * time_fraction
        
        # Add slight variation to other parameters (±2-3%)
        input_copy["Air temperature"] += np.random.uniform(-1, 1)
        input_copy["Process temperature"] += np.random.uniform(-1, 1)
        input_copy["Rotational speed"] += np.random.uniform(-50, 50)
        input_copy["Torque"] += np.random.uniform(-2, 2)
        
        risk_prob = model.predict_proba(input_copy)[0][1]
        
        results.append({
            "time_point": i,
            "failure_risk": risk_prob * 100,
            "tool_wear": input_copy["Tool wear"].values[0]
        })
    
    return pd.DataFrame(results)

def categorize_risk(risk_probability):
    """Determine risk category and recommendation."""
    if risk_probability < 0.20:
        return "LOW RISK", "#0f2f1b", "#dcfce7", "#22c55e", "Continue normal operation. Monitor tool wear periodically."
    elif risk_probability < 0.50:
        return "MEDIUM RISK", "#3a2f05", "#fef3c7", "#facc15", "Review operating conditions. Consider preventive maintenance schedule."
    else:
        return "HIGH RISK", "#3b0a0a", "#fee2e2", "#ef4444", "Flag for engineering review. Plan maintenance intervention."

def get_risk_driver_explanation(input_data, feature_importance_df, features):
    """
    Generate a human-readable explanation of risk drivers.
    """
    input_dict = input_data.iloc[0].to_dict()
    
    # Identify high-risk factors based on their values
    high_wear = input_dict["Tool wear"] > 150
    high_temp = input_dict["Process temperature"] > 312
    high_speed = input_dict["Rotational speed"] > 2000
    high_torque = input_dict["Torque"] > 60
    
    explanations = []
    
    if high_wear:
        explanations.append("**Tool wear** is elevated, increasing friction and risk of cascading failures.")
    
    if high_temp:
        explanations.append("**Process temperature** is high, which can accelerate thermal fatigue and material degradation.")
    
    if high_speed:
        explanations.append("**Rotational speed** is high, resulting in increased mechanical stress cycles.")
    
    if high_torque:
        explanations.append("**Torque** is significant, stressing bearings and load-bearing components.")
    
    if not explanations:
        explanations.append("Operating conditions are within normal ranges. Risk factors are well-controlled.")
    
    return " ".join(explanations)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Predictive Failure Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
    color: #f9fafb;
    padding: 30px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #4b5563;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.hero-header h1 {
    margin: 0;
    font-size: 2.5em;
}
.hero-header p {
    margin: 5px 0 0 0;
    font-size: 1.1em;
    opacity: 0.9;
}

/* Section Headers */
.section-header {
    color: #f9fafb;
    padding-bottom: 10px;
    border-bottom: 2px solid #667eea;
    margin-top: 20px;
    margin-bottom: 15px;
    font-weight: 700;
}

/* Metric Cards */
.metric-card {
    background: #1f2937;
    color: #f9fafb;
    padding: 18px 22px;
    border-radius: 10px;
    border-left: 6px solid #667eea;
    font-size: 1rem;
    margin-top: 12px;
    margin-bottom: 18px;
    line-height: 1.6;
}

.metric-card strong {
    color: #ffffff;
    font-size: 1.05rem;
}

.risk-low {
    background: #0f2f1b;
    color: #dcfce7;
    border-left-color: #22c55e;
}

.risk-medium {
    background: #3a2f05;
    color: #fef3c7;
    border-left-color: #facc15;
}

.risk-high {
    background: #3b0a0a;
    color: #fee2e2;
    border-left-color: #ef4444;
}

/* Info Cards */
.info-card {
    background: #1f2937;
    color: #e5e7eb;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #374151;
    margin: 12px 0;
    font-size: 0.95rem;
}

.info-card-title {
    color: #60a5fa;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Scenario Comparison */
.scenario-box {
    background: #111827;
    color: #e5e7eb;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #374151;
    margin: 10px 0;
}

.scenario-title {
    color: #93c5fd;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 12px;
}

.scenario-risk {
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 8px;
    color: #f0fdf4;
}

/* Chart container */
.chart-container {
    background: #111827;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #374151;
    margin: 15px 0;
}

/* Feature explanation */
.feature-explanation {
    background: #1f2937;
    color: #e5e7eb;
    padding: 14px;
    border-radius: 8px;
    border-left: 4px solid #60a5fa;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)
# ============================================================================
# LOAD MODEL AND METRICS
# ============================================================================
try:
    model = joblib.load("failure_model.pkl")
    features = joblib.load("model_features.pkl")
    
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    
    with open("feature_importance.json", "r") as f:
        feature_importance_data = json.load(f)
        feature_importance_df = pd.DataFrame({
            "Feature": feature_importance_data["features"],
            "Importance": feature_importance_data["importances"]
        }).sort_values("Importance", ascending=False)
    
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}\n\nPlease run training first:\n```\npython train_model.py\n```")
    st.stop()

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class="hero-header">
    <h1>Predictive Failure Risk Dashboard</h1>
    <p>Applied ML for engineering decision support and predictive maintenance</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: PROJECT INFO & CONTROLS
# ============================================================================
with st.sidebar:
    st.header("Dashboard Info")
    
    with st.expander("Project Overview", expanded=True):
        st.markdown("""
        This dashboard estimates machine failure risk from industrial operating 
        conditions using a trained RandomForest classifier.
        
        **Typical use cases:**
        - Predict failure risk before it impacts operations
        - Identify high-risk operating envelopes
        - Support maintenance planning decisions
        - Validate design margins under stress
        """)
    
    with st.expander("How to Interpret"):
        st.markdown("""
        **Risk Score:** Probability of failure (0–100%)
        
        **Risk Category:**
        - **Low Risk (<20%)**: Normal operation, standard monitoring
        - **Medium Risk (20–50%)**: Increase observation, plan maintenance
        - **High Risk (>50%)**: Urgent review, reduce operating stress
        
        **Confidence:** Model's certainty in this prediction
        """)
    
    with st.expander("Model Information"):
        st.markdown(f"""
        **Algorithm:** RandomForest Classifier (100 trees)
        
        **Test Accuracy:** {metrics['test_accuracy']:.1%}
        
        **Features:** {len(features)} operating conditions
        
        **Dataset:** UCI AI4I Predictive Maintenance (~10k samples)
        """)

# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Model Insights", "Engineering Interpretation", "About"])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================
with tab1:
    st.header("🎯 Risk Assessment")
    
    st.markdown("""
    Adjust operating conditions below to see real-time failure risk predictions and recommendations.
    """)
    
    # Operating conditions sliders
    col_sliders = st.columns(2)
    
    with col_sliders[0]:
        air_temp = st.slider(
            "Air Temperature [K]",
            min_value=295.0, max_value=305.0, value=300.0, step=0.5,
            help="Ambient temperature in Kelvin"
        )
        process_temp = st.slider(
            "Process Temperature [K]",
            min_value=305.0, max_value=315.0, value=310.0, step=0.5,
            help="Component/process temperature in Kelvin"
        )
        rot_speed = st.slider(
            "Rotational Speed [rpm]",
            min_value=1100, max_value=3000, value=1500, step=50,
            help="Spindle/motor speed in RPM"
        )
    
    with col_sliders[1]:
        torque = st.slider(
            "Torque [Nm]",
            min_value=3.0, max_value=80.0, value=40.0, step=1.0,
            help="Applied torque in Newton-meters"
        )
        tool_wear = st.slider(
            "Tool Wear [min]",
            min_value=0, max_value=260, value=100, step=10,
            help="Cumulative wear in minutes"
        )
    
    # Prepare current input
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
    confidence = max(model.predict_proba(input_data)[0])
    
    # Determine risk category
    risk_level, bg_color, text_color, border_color, recommendation = categorize_risk(risk_probability)
    
    # Display risk metrics
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        st.metric("Failure Risk", f"{risk_probability:.1%}", delta=None)
    
    with col_risk2:
        st.metric("Risk Category", risk_level)
    
    with col_risk3:
        st.metric("Model Confidence", f"{confidence:.1%}")
    
    # Recommendation box with dynamic styling
    st.markdown(f"""
    <div class='metric-card' style='background: {bg_color}; color: {text_color}; border-left-color: {border_color};'>
        <strong>Recommended Action:</strong><br>{recommendation}
    </div>
    """, unsafe_allow_html=True)
    
    # Current operating conditions
    st.markdown("<div class='section-header'>📋 Current Operating Conditions</div>", unsafe_allow_html=True)
    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )
    
    # ========================================================================
    # SCENARIO COMPARISON
    # ========================================================================
    st.markdown("<div class='section-header'>⚖️ Scenario Comparison</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>What is this?</div>
        Compare your current operating conditions against a conservative baseline scenario 
        to understand relative risk levels and potential improvements.
    </div>
    """, unsafe_allow_html=True)
    
    col_scenario1, col_scenario2 = st.columns(2)
    
    with col_scenario1:
        st.markdown("""
        <div class='scenario-box'>
            <div class='scenario-title'>🔴 Current Scenario</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Air Temperature:** {air_temp:.1f} K")
        st.write(f"**Process Temperature:** {process_temp:.1f} K")
        st.write(f"**Rotational Speed:** {rot_speed:.0f} rpm")
        st.write(f"**Torque:** {torque:.1f} Nm")
        st.write(f"**Tool Wear:** {tool_wear:.0f} min")
        
        st.markdown(f"""
        <div style='background: {bg_color}; color: {text_color}; padding: 12px; border-radius: 8px; margin-top: 10px; font-weight: 700;'>
        Risk: {risk_probability:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    with col_scenario2:
        st.markdown("""
        <div class='scenario-box'>
            <div class='scenario-title'>🟢 Conservative Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        
        baseline_data = pd.DataFrame([{
            "Air temperature": 298.0,
            "Process temperature": 308.0,
            "Rotational speed": 1200,
            "Torque": 30.0,
            "Tool wear": 50
        }])
        baseline_risk = model.predict_proba(baseline_data)[0][1]
        
        st.write(f"**Air Temperature:** 298.0 K")
        st.write(f"**Process Temperature:** 308.0 K")
        st.write(f"**Rotational Speed:** 1200 rpm")
        st.write(f"**Torque:** 30.0 Nm")
        st.write(f"**Tool Wear:** 50 min")
        
        st.markdown(f"""
        <div style='background: #0f2f1b; color: #dcfce7; padding: 12px; border-radius: 8px; margin-top: 10px; font-weight: 700;'>
        Risk: {baseline_risk:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Risk comparison
    risk_delta = (risk_probability - baseline_risk) * 100
    if risk_delta > 0:
        st.warning(f"⚠️ Current scenario is **{risk_delta:.1f}% higher risk** than conservative baseline. Consider reducing stress on critical components.")
    elif risk_delta < -5:
        st.success(f"✅ Current scenario is **{-risk_delta:.1f}% lower risk** than conservative baseline. Good operating margin.")
    else:
        st.info(f"ℹ️ Current scenario risk is comparable to conservative baseline (within {abs(risk_delta):.1f}%).")

# ============================================================================
# TAB 2: MODEL INSIGHTS
# ============================================================================
with tab2:
    st.header("Model Insights & Decision Support")
    
    st.markdown("""
    This section provides advanced analytical features to understand model behavior and make informed operational decisions.
    """)
    
    # ========================================================================
    # SENSITIVITY ANALYSIS
    # ========================================================================
    st.markdown("<div class='section-header'>📊 Sensitivity Analysis</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>What is this?</div>
        Sensitivity analysis shows how failure risk changes as a single operating variable 
        changes, while all other conditions remain fixed at their current values. This helps 
        you understand which variables have the biggest impact on predicted failure risk.
    </div>
    """, unsafe_allow_html=True)
    
    sensitivity_col1, sensitivity_col2 = st.columns([3, 1])
    
    with sensitivity_col2:
        variable_to_analyze = st.selectbox(
            "Analyze Variable",
            options=features,
            key="sensitivity_select"
        )
    
    with sensitivity_col1:
        st.markdown(f"*Showing how **{variable_to_analyze}** affects failure risk*")
    
    # Generate sensitivity data
    sensitivity_df = generate_sensitivity_analysis(model, features, input_data, variable_to_analyze)
    current_value = input_data[variable_to_analyze].values[0]
    
    # Create sensitivity chart
    fig_sens, ax_sens = plt.subplots(figsize=(11, 5))
    ax_sens.plot(
        sensitivity_df["variable_value"],
        sensitivity_df["failure_risk"],
        linewidth=2.5,
        color="#60a5fa",
        marker="o",
        markersize=4,
        markerfacecolor="#93c5fd"
    )
    
    # Add vertical line for current value
    ax_sens.axvline(current_value, color="#fbbf24", linestyle="--", linewidth=2, label="Current Value")
    
    # Add threshold lines for risk zones
    ax_sens.axhline(20, color="#22c55e", linestyle=":", linewidth=1.5, alpha=0.6, label="Medium Risk Threshold (20%)")
    ax_sens.axhline(50, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.6, label="High Risk Threshold (50%)")
    
    ax_sens.set_xlabel(f"{variable_to_analyze}", fontsize=11, fontweight="600")
    ax_sens.set_ylabel("Predicted Failure Risk (%)", fontsize=11, fontweight="600")
    ax_sens.set_title(f"Sensitivity: {variable_to_analyze} Impact on Failure Risk", fontsize=12, fontweight="bold")
    ax_sens.grid(True, alpha=0.2)
    ax_sens.legend(loc="best", fontsize=9)
    ax_sens.set_ylim(0, 100)
    
    plt.tight_layout()
    st.pyplot(fig_sens)
    
    st.markdown(f"""
    <div class='feature-explanation'>
    <strong>Interpretation:</strong> This chart shows how sensitive the model is to changes in 
    <strong>{variable_to_analyze}</strong> while keeping all other operating conditions constant. 
    The steeper the line, the more impactful this variable is on failure risk. The yellow dashed 
    line marks your current {variable_to_analyze} value.
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # RISK TREND SIMULATION
    # ========================================================================
    st.markdown("<div class='section-header'>📈 Risk Trend Simulation (Illustrative Demo)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>⚠️ Demo Data</div>
        This is <strong>synthetic illustrative data</strong>, not real field telemetry or Gates product data. 
        It demonstrates how a production dashboard could monitor and visualize risk trends over time.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-explanation'>
    This simulation shows how failure risk could evolve over time as tool wear gradually accumulates 
    and other operating conditions vary slightly. In a production system, this would use real sensor 
    data and historical observations.
    </div>
    """, unsafe_allow_html=True)
    
    # Generate trend data (seeded for reproducibility)
    np.random.seed(42)
    trend_df = generate_risk_trend_demo(model, features, input_data, num_points=25)
    
    # Create trend chart
    fig_trend, ax_trend = plt.subplots(figsize=(11, 5))
    
    ax_trend.plot(
        trend_df["time_point"],
        trend_df["failure_risk"],
        linewidth=2.5,
        color="#8b5cf6",
        marker="o",
        markersize=5,
        markerfacecolor="#c4b5fd",
        label="Predicted Failure Risk"
    )
    
    # Add risk threshold zones
    ax_trend.axhspan(0, 20, alpha=0.1, color="#22c55e", label="Low Risk Zone")
    ax_trend.axhspan(20, 50, alpha=0.1, color="#facc15", label="Medium Risk Zone")
    ax_trend.axhspan(50, 100, alpha=0.1, color="#ef4444", label="High Risk Zone")
    
    ax_trend.set_xlabel("Time Points (Simulated)", fontsize=11, fontweight="600")
    ax_trend.set_ylabel("Predicted Failure Risk (%)", fontsize=11, fontweight="600")
    ax_trend.set_title("Risk Trend Over Time (Synthetic Demo)", fontsize=12, fontweight="bold")
    ax_trend.grid(True, alpha=0.2)
    ax_trend.legend(loc="best", fontsize=9)
    ax_trend.set_ylim(0, 100)
    
    plt.tight_layout()
    st.pyplot(fig_trend)
    
    st.markdown(f"""
    <div class='feature-explanation'>
    <strong>What this shows:</strong> As time progresses, tool wear accumulates (from {trend_df.iloc[0]['tool_wear']:.1f} min 
    to {trend_df.iloc[-1]['tool_wear']:.1f} min), and the model predicts failure risk increases. 
    The risk crosses into the medium zone around time point {trend_df[trend_df['failure_risk'] > 20].iloc[0]['time_point'] if any(trend_df['failure_risk'] > 20) else 'N/A'}, 
    indicating when preventive maintenance should be scheduled.
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # Feature Importance
    # ========================================================================
    st.markdown("<div class='section-header'>⚙️ Feature Importance in Prediction</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>What is this?</div>
        Feature importance shows which operating conditions have the strongest influence on 
        failure risk predictions. Higher values indicate variables with more predictive power.
    </div>
    """, unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11, fontweight="600")
    ax.set_title("Which Operating Conditions Drive Model Predictions?", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(feature_importance_df, use_container_width=True)
    
    # ========================================================================
    # Risk Drivers for Current Prediction
    # ========================================================================
    st.markdown("<div class='section-header'>🎯 Risk Drivers (Current Prediction)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>What is this?</div>
        This combines current input values with feature importance to identify which factors 
        are contributing most to your predicted failure risk.
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate risk contributions
    input_dict = input_data.iloc[0].to_dict()
    
    ranges = {
        "Air temperature": (295, 305),
        "Process temperature": (305, 315),
        "Rotational speed": (1100, 3000),
        "Torque": (3, 80),
        "Tool wear": (0, 260)
    }
    
    driver_contributions = {}
    for feat in features:
        min_val, max_val = ranges[feat]
        normalized = (input_dict[feat] - min_val) / (max_val - min_val)
        importance = feature_importance_df[feature_importance_df["Feature"] == feat]["Importance"].values[0]
        driver_contributions[feat] = max(0, normalized * importance)
    
    driver_df = pd.DataFrame({
        "Feature": driver_contributions.keys(),
        "Risk Contribution": driver_contributions.values()
    }).sort_values("Risk Contribution", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_drivers = ["#dc3545" if x > 0.1 else "#ffc107" if x > 0.05 else "#28a745" 
                      for x in driver_df["Risk Contribution"]]
    ax.barh(driver_df["Feature"], driver_df["Risk Contribution"], color=colors_drivers)
    ax.set_xlabel("Relative Risk Contribution", fontsize=11, fontweight="600")
    ax.set_title("Which Current Input Values Are Driving Risk?", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Engineering explanation of risk drivers
    driver_explanation = get_risk_driver_explanation(input_data, feature_importance_df, features)
    
    st.markdown(f"""
    <div class='feature-explanation'>
    <strong>Based on your current operating conditions:</strong><br>
    {driver_explanation}
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # Model Performance Metrics
    # ========================================================================
    st.markdown("<div class='section-header'>📊 Model Performance (Test Set)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div class='info-card-title'>What is this?</div>
        These metrics show how well the trained model performed on test data it had never seen during training.
    </div>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy", f"{metrics['test_accuracy']:.1%}")
    col_m2.metric("Precision", f"{metrics['precision']:.1%}")
    col_m3.metric("Recall", f"{metrics['recall']:.1%}")
    col_m4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
    
    st.markdown("**Confusion Matrix** (Test Set Predictions):")
    cm = np.array(metrics['confusion_matrix'])
    
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted: No Failure", "Predicted: Failure"],
        index=["Actual: No Failure", "Actual: Failure"]
    )
    st.dataframe(cm_df, use_container_width=True)
    
    st.markdown("""
    <div class='feature-explanation'>
    <strong>Interpretation Guide:</strong><br>
    • <strong>True Negatives:</strong> Correctly identified non-failures (good)<br>
    • <strong>False Positives:</strong> Over-cautious predictions (conservative)<br>
    • <strong>False Negatives:</strong> Missed failures (risky in practice)<br>
    • <strong>True Positives:</strong> Correctly identified failures (good)
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: ENGINEERING INTERPRETATION
# ============================================================================
with tab3:
    st.header("🔧 Engineering Interpretation & Failure Modes")
    
    st.markdown("""
    This section translates model predictions into engineering language, connecting 
    machine learning outputs to real failure mechanisms. This is interpretive analysis 
    based on the operating variables—not actual Gates product data or proprietary information.
    """)
    
    # ========================================================================
    # Failure Mode Mapping
    # ========================================================================
    st.markdown("<div class='section-header'>⚡ Potential Failure Mode Pathways</div>", unsafe_allow_html=True)
    
    col_modes1, col_modes2 = st.columns(2)
    
    with col_modes1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>🔥 Thermal Stress Pathway</div>
        </div>
        """, unsafe_allow_html=True)
        
        temp_risk = "High" if process_temp > 312 else "Moderate" if process_temp > 310 else "Low"
        st.markdown(f"""
        **Current Air/Process Temps:** {air_temp:.1f} K / {process_temp:.1f} K
        
        - Thermal cycling induces material fatigue
        - Lubrication breakdown accelerates at high temperatures  
        - Seal and gasket material degradation over time
        - Differential expansion stresses joints and interfaces
        
        **Assessment:** {temp_risk}
        """)
        
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>⚙️ Mechanical Stress Pathway</div>
        </div>
        """, unsafe_allow_html=True)
        
        stress_risk = "High" if (rot_speed > 2000 or torque > 60) else "Moderate" if (rot_speed > 1500 or torque > 40) else "Low"
        st.markdown(f"""
        **Current Speed/Torque:** {rot_speed:.0f} rpm / {torque:.1f} Nm
        
        - High rotational speed increases fatigue cycles per unit time
        - High torque stresses bearings, shafts, and drive interfaces
        - Combined stress accelerates progressive wear
        - Resonance risks increase at certain speed/load combinations
        
        **Assessment:** {stress_risk}
        """)
    
    with col_modes2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>🛠️ Wear & Degradation Pathway</div>
        </div>
        """, unsafe_allow_html=True)
        
        wear_risk = "High" if tool_wear > 150 else "Moderate" if tool_wear > 75 else "Low"
        st.markdown(f"""
        **Current Tool Wear:** {tool_wear:.0f} min
        
        - Progressive wear increases friction and heat generation
        - Worn surfaces lose geometric precision and load distribution
        - Wear particles can trigger cascading micro-failures
        - Tool life limits are typically defined by this metric
        - Preventive replacement windows should align with wear thresholds
        
        **Assessment:** {wear_risk}
        """)
        
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>🌡️ Combined Stress Scenario</div>
        </div>
        """, unsafe_allow_html=True)
        
        combined_risk_factors = sum([
            process_temp > 312,
            rot_speed > 2000 or torque > 60,
            tool_wear > 150
        ])
        
        st.markdown(f"""
        **Risk Factors Active:** {combined_risk_factors} / 3
        
        When multiple stress pathways activate simultaneously, failure risk 
        increases more rapidly due to interaction effects and reduced safety margins.
        """)
    
    # ========================================================================
    # Engineering Recommendations
    # ========================================================================
    st.markdown("<div class='section-header'>💡 Engineering Recommendations</div>", unsafe_allow_html=True)
    
    if risk_probability < 0.20:
        st.success("""
        ### ✅ Low Risk Zone – Continue Normal Operation
        
        **Operational Guidance:**
        - Continue standard operating procedures
        - Maintain normal monitoring intervals
        - Document baseline conditions for trend analysis
        - Track tool wear progression over time
        - No immediate maintenance action required
        
        **Validation & Testing:**
        - Current conditions validate design margins
        - Consider this a safe operating envelope
        - Good candidate for baseline performance data collection
        """)
    
    elif risk_probability < 0.50:
        st.warning("""
        ### ⚠️ Medium Risk Zone – Plan Maintenance Window
        
        **Operational Guidance:**
        - Increase monitoring frequency and detail
        - Consider gradual reduction of speed or load if feasible
        - Schedule preventive maintenance within current operating cycle
        - Improve thermal management or cooling system performance
        - Prepare replacement components for rapid swap
        
        **Recommended Actions:**
        1. Collect detailed telemetry over next 24–48 hours
        2. Schedule maintenance inspection before next cycle
        3. Identify backup hardware if mission-critical
        4. Review and verify thermal management adequacy
        5. Document conditions that triggered this alert
        
        **Testing & Validation:**
        - Use this as an input case study for durability analysis
        - Collect field or test data to validate model predictions
        """)
    
    else:
        st.error("""
        ### 🚨 High Risk Zone – Urgent Action Required
        
        **Immediate Actions:**
        1. **REDUCE OPERATING STRESS** immediately
           - Decrease rotational speed if possible
           - Lower applied torque or load
           - Reduce process temperature or improve cooling
        2. **Schedule immediate maintenance inspection**
        3. **Prepare for hardware replacement**
        4. **Escalate to engineering/reliability team**
        
        **Operational Guidance:**
        - Do not continue extended operation at current settings
        - Collect detailed failure mode diagnostics
        - Inspect visually for wear, cracks, thermal damage
        - Review recent telemetry for anomalies
        - Prepare comprehensive incident report
        
        **Testing & Validation:**
        - Transition to backup hardware if available
        - Gather failure data for post-mortem root cause analysis
        - Use this case to refine predictive model and thresholds
        """)
    
    # ========================================================================
    # Gates Relevance
    # ========================================================================
    st.markdown("<div class='section-header'>🏭 Applied ML in Engineering – Use Cases</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard demonstrates how predictive analytics supports real engineering workflows:
    
    #### 1. **Predictive Durability Modeling**
    - Estimate component/product life under different operating envelopes
    - Predict service life from field operating conditions
    - Identify which stress factors dominate failures
    
    #### 2. **Design Validation & Margins**
    - Validate design margins against predicted failure risk
    - Identify which design parameters need improvement
    - Prioritize design improvements with data-driven analysis
    
    #### 3. **Accelerated Testing & Simulation**
    - Guide accelerated life test (ALT) parameter selection
    - Predict failure points without running full tests
    - Optimize test duration and stress levels
    
    #### 4. **Field Maintenance & Support**
    - Alert field engineers to emerging failure risk
    - Recommend proactive maintenance and replacement intervals
    - Reduce unexpected downtime and warranty costs
    - Enable condition-based maintenance (CBM) strategies
    
    #### 5. **Decision Support & Trade-Off Analysis**
    - Quantify reliability vs. performance trade-offs
    - Provide data-driven recommendations to stakeholders
    - Enable speed vs. safety comparisons with quantified risk
    - Support cost/benefit analysis of design alternatives
    """)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab4:
    st.header("📚 About This Project")
    
    st.markdown("""
    ## Predictive Failure Risk Dashboard
    
    A demonstration of applied machine learning for engineering decision support 
    in predictive maintenance and industrial systems.
    
    ---
    
    ## Project Vision
    
    This dashboard is not a production system or Gates product. Instead, it demonstrates 
    how predictive analytics can transform raw model outputs into actionable 
    engineering insights. The goal is to show thoughtful systems design: 
    not just *accuracy*, but *usability* for engineers making real decisions.
    
    **Key Design Decisions:**
    - **Sensitivity Analysis:** Engineers need to understand which variables matter most
    - **Scenario Comparison:** Decisions are relative—compare against a baseline
    - **Risk Trend Simulation:** Show how to monitor risk over time (not just one-off predictions)
    - **Engineering Interpretation:** Translate ML outputs to failure modes engineers recognize
    - **Honest Framing:** Be clear about what is demo data vs. real product data
    
    ---
    
    """)
    
    # Dataset Info
    col_ds1, col_ds2 = st.columns(2)
    
    with col_ds1:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>📊 Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Source:** UCI AI4I 2020 Predictive Maintenance Dataset
        
        **Records:** ~10,000 synthetic operating samples
        
        **Features:**
        - Air temperature (K)
        - Process temperature (K)
        - Rotational speed (rpm)
        - Torque (Nm)
        - Tool wear (min)
        
        **Target:** Binary machine failure classification
        
        **Why This Dataset?** While synthetic, it mirrors real-world 
        industrial telemetry (thermocouples, tachometers, load cells, 
        wear sensors).
        """)
    
    with col_ds2:
        st.markdown("""
        <div class='info-card'>
            <div class='info-card-title'>🤖 Model Details</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Algorithm:** RandomForest Classifier
        
        **Configuration:**
        - 100 decision trees
        - Balanced class weights (handles ~15% failure rate)
        - Test accuracy: {metrics['test_accuracy']:.1%}
        
        **Why RandomForest?**
        - Fast inference (production-ready)
        - Feature importance is directly interpretable
        - Handles non-linear relationships
        - Robust to class imbalance
        - No special dependencies beyond scikit-learn
        
        **Top Predictors:**
        """)
        
        for idx, row in feature_importance_df.head(3).iterrows():
            st.markdown(f"- **{row['Feature']}** ({row['Importance']:.1%})")
    
    # Test Performance
    st.markdown("""
    <div class='info-card' style='margin-top: 20px;'>
        <div class='info-card-title'>📈 Model Performance (Test Set)</div>
    </div>
    """, unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Accuracy", f"{metrics['test_accuracy']:.1%}")
    perf_col2.metric("Precision", f"{metrics['precision']:.1%}")
    perf_col3.metric("Recall", f"{metrics['recall']:.1%}")
    perf_col4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
    
    # Key Insights
    st.markdown("""
    <div class='info-card' style='margin-top: 20px;'>
        <div class='info-card-title'>💡 Key Insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    **Most Important Factors:**
    - Tool wear drives failure risk more than any other factor
    - High rotational speed and torque create cumulative mechanical stress
    - Temperature effects are significant but secondary to wear
    
    **Implication for Engineering:**
    - Maintenance schedules should prioritize wear monitoring
    - Speed/load management is critical for reliability
    - Thermal management is important but not the dominant failure driver
    
    **Data Pattern:**
    - Failure events cluster in high-wear, high-stress scenarios
    - Failures are relatively rare (~15% of dataset) but predictable
    - Early identification of high-risk conditions is feasible
    """)
    
    # Limitations
    st.markdown("""
    <div class='info-card' style='margin-top: 20px;'>
        <div class='info-card-title'>⚠️ Important Limitations</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ✗ **Not production software** – For demonstration only  
    ✗ **Not based on real Gates data** – Synthetic dataset for learning  
    ✗ **Not replacing physical testing** – Use alongside validation programs  
    ✗ **No proprietary models** – Simple open-source algorithms  
    ✗ **Binary predictions only** – Doesn't distinguish failure modes  
    ✗ **No uncertainty quantification** – Point estimates, not confidence intervals  
    """)
    
    # Future Work
    st.markdown("""
    <div class='info-card' style='margin-top: 20px;'>
        <div class='info-card-title'>🚀 Future Enhancements</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Near-term:**
    - SHAP values for per-prediction feature explanations
    - Remaining Useful Life (RUL) estimation (time-series forecasting)
    - Multi-class failure mode prediction
    - Unit test coverage for data pipeline
    
    **Medium-term:**
    - Real product test or field data integration
    - Continuous retraining workflow
    - API backend for manufacturing system integration
    - Support for multiple product/asset types
    
    **Long-term:**
    - Multi-model ensembles (RF + XGBoost + neural networks)
    - Bayesian uncertainty quantification
    - Causal inference (move beyond correlation)
    - Docker containerization and cloud deployment (AWS/Azure)
    - Model monitoring and drift detection
    """)
    
    st.divider()
    
    st.markdown("""
    **Questions or feedback?** This is a learning project. 
    See the [GitHub repository](https://github.com/your-repo/predictive-failure-risk-dashboard) 
    for source code and more details.
    
    ---
    *A demonstration of thoughtful applied ML for engineering decision support.*
    """)
