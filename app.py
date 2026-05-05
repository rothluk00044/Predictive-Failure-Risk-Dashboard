"""
app.py

Streamlit dashboard for predictive failure risk assessment.
Demonstrates applied ML for engineering decision support in industrial settings.

Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

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

    .metric-card {
        background: #1f2937;
        color: #f9fafb;
        padding: 18px 22px;
        border-radius: 10px;
        border-left: 6px solid #667eea;
        font-size: 1rem;
        margin-top: 12px;
        margin-bottom: 18px;
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
    st.header("Real-Time Risk Assessment")
    
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
    if risk_probability < 0.20:
        risk_color = "green"
        risk_level = "LOW RISK"
        recommendation = "Continue normal operation. Monitor tool wear periodically."
        risk_class = "risk-low"
    elif risk_probability < 0.50:
        risk_color = "orange"
        risk_level = "MEDIUM RISK"
        recommendation = "Review operating conditions. Consider preventive maintenance schedule."
        risk_class = "risk-medium"
    else:
        risk_color = "red"
        risk_level = "HIGH RISK"
        recommendation = "Flag for engineering review. Plan maintenance intervention."
        risk_class = "risk-high"
    
    # Display risk in large cards
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        st.metric("Failure Risk", f"{risk_probability:.1%}", delta=None)
    
    with col_risk2:
        st.metric("Risk Category", risk_level)
    
    with col_risk3:
        st.metric("Model Confidence", f"{confidence:.1%}")
    
    # Recommendation box
    st.markdown(f"<div class='metric-card {risk_class}'><strong>Recommended Action:</strong><br>{recommendation}</div>", 
                unsafe_allow_html=True)
    
    # Current operating conditions
    st.subheader("Current Operating Conditions")
    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )
    
    # ========================================================================
    # SCENARIO COMPARISON
    # ========================================================================
    st.subheader("Scenario Comparison")
    st.markdown("Compare current conditions against a safer baseline.")
    
    col_scenario1, col_scenario2 = st.columns(2)
    
    with col_scenario1:
        st.write("**Current Scenario**")
        st.write(f"- Air Temp: {air_temp:.1f} K")
        st.write(f"- Process Temp: {process_temp:.1f} K")
        st.write(f"- Speed: {rot_speed:.0f} rpm")
        st.write(f"- Torque: {torque:.1f} Nm")
        st.write(f"- Wear: {tool_wear:.0f} min")
        current_risk_display = f"**Risk: {risk_probability:.1%}**"
        st.write(current_risk_display)
    
    with col_scenario2:
        st.write("**Conservative Baseline** (Lower Stress)")
        baseline_data = pd.DataFrame([{
            "Air temperature": 298.0,
            "Process temperature": 308.0,
            "Rotational speed": 1200,
            "Torque": 30.0,
            "Tool wear": 50
        }])
        baseline_risk = model.predict_proba(baseline_data)[0][1]
        
        st.write(f"- Air Temp: 298.0 K")
        st.write(f"- Process Temp: 308.0 K")
        st.write(f"- Speed: 1200 rpm")
        st.write(f"- Torque: 30.0 Nm")
        st.write(f"- Wear: 50 min")
        baseline_risk_display = f"**Risk: {baseline_risk:.1%}**"
        st.write(baseline_risk_display)
    
    risk_delta = (risk_probability - baseline_risk) * 100
    if risk_delta > 0:
        st.warning(f"Current scenario is **{risk_delta:.1f}% higher risk** than baseline. Consider stress reduction.")
    else:
        st.success(f"Current scenario is **{-risk_delta:.1f}% lower risk** than baseline.")

# ============================================================================
# TAB 2: MODEL INSIGHTS
# ============================================================================
with tab2:
    st.header("Model Insights & Analysis")
    
    # ========================================================================
    # Feature Importance
    # ========================================================================
    st.subheader("Feature Importance")
    st.markdown("""
    Shows which operating conditions have the strongest influence on failure risk 
    (based on how much each feature contributes to model decisions during training).
    """)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title("Feature Importance in Failure Risk Prediction", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(feature_importance_df, use_container_width=True)
    
    # ========================================================================
    # Risk Drivers for Current Prediction
    # ========================================================================
    st.subheader("Risk Drivers (Current Prediction)")
    st.markdown("""
    Which inputs are likely contributing to the current risk level?
    """)
    
    # Create a simple explanation based on feature importance and current values
    input_dict = input_data.iloc[0].to_dict()
    
    # Normalize inputs to 0-1 scale for rough risk contribution
    driver_contributions = {}
    
    ranges = {
        "Air temperature": (295, 305),
        "Process temperature": (305, 315),
        "Rotational speed": (1100, 3000),
        "Torque": (3, 80),
        "Tool wear": (0, 260)
    }
    
    for feat in features:
        min_val, max_val = ranges[feat]
        normalized = (input_dict[feat] - min_val) / (max_val - min_val)
        importance = feature_importance_df[feature_importance_df["Feature"] == feat]["Importance"].values[0]
        driver_contributions[feat] = normalized * importance
    
    driver_df = pd.DataFrame({
        "Feature": driver_contributions.keys(),
        "Risk Contribution": driver_contributions.values()
    }).sort_values("Risk Contribution", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_drivers = ["#dc3545" if x > 0.1 else "#ffc107" if x > 0.05 else "#28a745" 
                      for x in driver_df["Risk Contribution"]]
    ax.barh(driver_df["Feature"], driver_df["Risk Contribution"], color=colors_drivers)
    ax.set_xlabel("Relative Risk Contribution", fontsize=11)
    ax.set_title("Which Inputs Are Driving Risk in Current Scenario?", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    # ========================================================================
    # Model Performance Metrics
    # ========================================================================
    st.subheader("Model Performance (Test Set)")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuracy", f"{metrics['test_accuracy']:.1%}")
    col_m2.metric("Precision", f"{metrics['precision']:.1%}")
    col_m3.metric("Recall", f"{metrics['recall']:.1%}")
    col_m4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
    
    st.markdown("**Confusion Matrix** (Test predictions):")
    cm = np.array(metrics['confusion_matrix'])
    
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted: No Failure", "Predicted: Failure"],
        index=["Actual: No Failure", "Actual: Failure"]
    )
    st.dataframe(cm_df, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - True Negatives: Correctly identified non-failures
    - False Positives: Over-cautious (flagged as failure but didn't)
    - False Negatives: Missed failures (risky in practice)
    - True Positives: Correctly identified failures
    """)

# ============================================================================
# TAB 3: ENGINEERING INTERPRETATION
# ============================================================================
with tab3:
    st.header("Engineering Perspective & Failure Modes")
    
    st.markdown("""
    This section maps the model's binary failure prediction to potential 
    real-world failure mechanisms. This is interpretive framing based on 
    the operating variables in the model—not actual Gates product data.
    """)
    
    # ========================================================================
    # Failure Mode Mapping
    # ========================================================================
    st.subheader("Potential Failure Mode Pathways")
    
    col_modes1, col_modes2 = st.columns(2)
    
    with col_modes1:
        st.markdown("#### Temperature-Related Stress")
        st.markdown(f"""
        **Current air/process temps:** {air_temp:.1f} K / {process_temp:.1f} K
        
        - Thermal cycling can cause material fatigue
        - Lubrication breakdown at high temperatures
        - Seal/gasket degradation
        
        **Risk factor:** {'High' if process_temp > 312 else 'Moderate' if process_temp > 310 else 'Low'}
        """)
        
        st.markdown("#### Speed & Load Stress")
        st.markdown(f"""
        **Current speed/torque:** {rot_speed:.0f} rpm / {torque:.1f} Nm
        
        - High-speed operation increases fatigue cycles
        - High torque stresses bearings and shafts
        - Combined stress accelerates wear
        
        **Risk factor:** {'High' if rot_speed > 2000 or torque > 60 else 'Moderate' if rot_speed > 1500 or torque > 40 else 'Low'}
        """)
    
    with col_modes2:
        st.markdown("#### Wear & Degradation")
        st.markdown(f"""
        **Current tool wear:** {tool_wear:.0f} min
        
        - Progressive wear increases friction
        - Worn surfaces lose geometric precision
        - Wear particles can trigger cascading failures
        - Maintenance windows should be planned around wear thresholds
        
        **Risk factor:** {'High' if tool_wear > 150 else 'Moderate' if tool_wear > 75 else 'Low'}
        """)
    
    # ========================================================================
    # Engineering Recommendations
    # ========================================================================
    st.subheader("Engineering Recommendations")
    
    if risk_probability < 0.20:
        st.success("""
        **LOW RISK OPERATING ZONE**
        
        - Continue normal operation
        - Maintain standard monitoring intervals
        - Track tool wear trends over time
        - Document normal operating baselines
        """)
    elif risk_probability < 0.50:
        st.warning("""
        **MEDIUM RISK - REVIEW REQUIRED**
        
        - Increase monitoring frequency
        - Consider reducing speed or load if possible
        - Plan preventive maintenance within next operating cycle
        - Review thermal management / cooling adequacy
        - Prepare replacement hardware for quick swap
        """)
    else:
        st.error("""
        **HIGH RISK - URGENT ACTION**
        
        - **Reduce operating stress immediately** (lower speed, torque, or temperature)
        - Schedule maintenance intervention before next cycle
        - Inspect for visible wear, degradation, or anomalies
        - Consider switching to backup hardware if available
        - Collect detailed telemetry to understand failure progression
        """)
    
    # ========================================================================
    # Gates Relevance
    # ========================================================================
    st.subheader("Gates Corporation Use Cases")
    
    st.markdown("""
    **How predictive maintenance applies to Gates products:**
    
    1. **Predictive Performance Modeling**
       - Estimate durability under different operating envelopes
       - Predict belt/hose life from field operating conditions
       
    2. **Failure Mode Assessment**
       - Identify which stress factors (temperature, speed, load) dominate failures
       - Validate design margins against test/field data
       
    3. **Testing & Validation**
       - Guide accelerated life testing parameters
       - Predict when prototypes will fail under stress
       - Optimize test schedules
       
    4. **Maintenance Planning**
       - Alert field engineers to emerging failure risk
       - Recommend proactive replacement intervals
       - Reduce unexpected downtime
       
    5. **Decision Support**
       - Quantify risk in terms engineers understand
       - Provide data-driven recommendations
       - Enable trade-off analysis (e.g., speed vs. reliability)
    """)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab4:
    st.header("About This Project")
    
    st.markdown("""
    ## Predictive Failure Risk Dashboard
    
    A demonstration of applied machine learning for engineering decision support 
    in industrial settings.
    
    ### Dataset
    - **Source:** UCI AI4I 2020 Predictive Maintenance Dataset
    - **Size:** ~10,000 synthetic operating records
    - **Features:** Air temperature, process temperature, rotational speed, torque, tool wear
    - **Target:** Binary machine failure classification
    
    ### Model
    - **Algorithm:** RandomForest Classifier (100 trees)
    - **Training:** 80% train / 20% test split
    - **Class Balancing:** Yes (to handle ~15% failure rate)
    - **Test Accuracy:** {metrics['test_accuracy']:.1%}
    
    ### Key Insights
    - **Top predictor:** {feature_importance_df.iloc[0]['Feature']} ({feature_importance_df.iloc[0]['Importance']:.1%})
    - **Most important factors:** Tool wear, rotational speed, torque stress
    - **Implication:** Wear and mechanical stress dominate failure risk
    
    ### What This Is NOT
    - Not production-grade software
    - Not based on real Gates product data
    - Not replacing physical testing or validation
    
    ### Future Improvements
    - Integrate real product test/field data
    - Predict specific failure modes instead of binary
    - Add time-series trend modeling (RUL estimation)
    - Implement SHAP for per-prediction explainability
    - Add design optimization recommendations
    - Deploy on cloud platform (Streamlit Cloud, AWS, etc.)
    """)
    
    st.markdown("---")
    st.markdown("""
    **GitHub:** [Predictive-Failure-Risk-Dashboard](https://github.com/rothluk00044/Predictive-Failure-Risk-Dashboard)
    """)
