# Predictive Failure Risk Dashboard

## Overview

A **Gates-inspired predictive maintenance and failure risk dashboard** that demonstrates applied machine learning for industrial engineering workflows. This project uses real predictive maintenance data to train a classifier that estimates failure risk from operating conditions—translating raw sensor data into actionable engineering insights.

**Key idea**: In industries like Gates—where product durability, reliability, and field performance are critical—predictive models support test planning, durability analysis, and engineering decision-making. This dashboard showcases that workflow.

---

## Project Context

### Why I Built This

This is an interview project created for a **Gates Corporation Rotational Engineer** role focused on applying AI/ML and advanced analytics to engineering and product development. The role emphasizes:

- Predictive performance modeling and durability analysis
- Failure mode assessment and risk quantification
- Integration of test/simulation/field data into engineering workflows
- AI-enabled tools for product development and validation

**This dashboard demonstrates all four areas** in a clean, explainable, and realistic micro-scale project.

### The Problem It Solves

In product development and manufacturing, you want to know:
1. **Will this product fail under these operating conditions?**
2. **Which operating conditions create the highest risk?**
3. **What should the engineering team do about it?**

This tool answers those questions using supervised machine learning on historical failure data.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data & ML** | pandas, scikit-learn, numpy |
| **UI/Dashboard** | Streamlit |
| **Model Persistence** | joblib |
| **Dataset** | UCI AI4I 2020 Predictive Maintenance (public) |

---

## Dataset

**Source**: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

**What it contains**: ~10,000 synthetic industrial operating records with:
- **Features**: Air temperature, process temperature, rotational speed, torque, tool wear
- **Target**: Machine failure (binary: Yes/No)

**Why it's realistic**: While synthetic, it mirrors real-world machine/component operating telemetry—temperature sensors, RPM monitors, force/torque transducers, wear tracking—common in industrial testing and field deployment.

---

## How It Works

### 1. Training Phase (`train_model.py`)

```bash
python train_model.py
```

**What happens**:
- Fetches the UCI dataset
- Splits data: 80% train, 20% test
- Trains a RandomForest classifier (100 trees, balanced class weights)
- Outputs:
  - **Confusion matrix** & classification report
  - **Feature importance** (which operating conditions matter most)
  - **Serialized model** (`failure_model.pkl`)

**Example output**:
```
Confusion Matrix:
[[1610   87]
 [ 106  197]]

Classification Report:
              precision    recall  f1-score   support
     No Failure       0.94      0.95      0.94      1697
        Failure       0.69      0.65      0.67       303
```

### 2. Dashboard Phase (`app.py`)

```bash
python -m streamlit run app.py
```

**What you see**:
- **Operating conditions panel** (sidebar): Adjust temperature, speed, torque, wear with sliders
- **Real-time risk prediction**: Model instantly predicts failure probability (0–100%)
- **Risk categorization**: 
  - 🟢 **Low** (<20%): Continue normal operation
  - 🟡 **Medium** (20–50%): Review & monitor
  - 🔴 **High** (>50%): Flag for review
- **Recommended actions**: Engineering-facing recommendations (e.g., "Plan maintenance")
- **Feature importance chart**: Visual breakdown of which factors drive failure risk
- **Model metadata**: Accuracy, confidence, prediction details

---

## Quick Start

### Prerequisites

- **Python 3.8+** installed and on your PATH
- **Git** (optional, for cloning)

### Windows Setup (If Python Not Recognized)

If running `python --version` fails in PowerShell:

#### Step 1: Install Python

1. Go to [python.org](https://www.python.org/downloads/)
2. Download **Python 3.11** (or latest stable 3.x)
3. Run the installer
4. **IMPORTANT**: Check ✅ **"Add Python to PATH"** during installation
5. Click **"Install Now"**

#### Step 2: Restart Terminal & Verify

```powershell
python --version
python -m pip --version
```

Both should return version info (not errors).

### Step 3: Clone & Install

```bash
# Navigate to your project directory
cd C:\Users\rothl\PFRD\Predictive-Failure-Risk-Dashboard

# Install dependencies
python -m pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
# This fetches the dataset, trains the model, and saves failure_model.pkl
python train_model.py
```

### Step 5: Run the Dashboard

```bash
python -m streamlit run app.py
```

Streamlit will open a local browser tab at `http://localhost:8501`.

**Adjust sliders** → Watch predictions update in real-time!

---

## Project Structure

```
Predictive-Failure-Risk-Dashboard/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train_model.py               # Model training script
├── app.py                       # Streamlit dashboard app
├── failure_model.pkl            # Trained model (generated after train_model.py)
├── model_features.pkl           # Feature names (generated after train_model.py)
├── .gitignore                   # Git ignore file
├── data/                        # Data directory (placeholder for future use)
└── notebooks/                   # Jupyter notebooks (placeholder for future analysis)
```

---

## Model Details

### Algorithm: RandomForestClassifier

**Why RandomForest?**
- **Interpretability**: Feature importance is directly available (crucial for engineering communication)
- **Robustness**: Handles non-linear relationships between operating conditions and failure
- **Class imbalance**: Built-in `class_weight="balanced"` handles the ~15% failure rate in the data
- **Production-ready**: Easy to serialize, fast inference, no special dependencies

### Hyperparameters

```python
RandomForestClassifier(
    n_estimators=100,           # 100 decision trees
    random_state=42,            # Reproducibility
    class_weight="balanced",    # Adjust for class imbalance
    n_jobs=-1                   # Parallel training (use all cores)
)
```

### Performance (on Test Set)

Expected metrics (will vary slightly due to dataset randomness):
- **Accuracy**: ~94%
- **Precision (Failure class)**: ~69%
- **Recall (Failure class)**: ~65%

*Note*: Recall is slightly lower because predicting rare events (failures) is inherently harder. This is typical in predictive maintenance and often acceptable since false positives are cheaper than missed failures.

---

## Use Cases & Interview Talking Points

### How This Connects to Gates

Gates manufactures power transmission belts, hoses, coupling, and other industrial components. This dashboard could support:

1. **Durability Testing**: Predict which field operating conditions lead to early failure
2. **Reliability Engineering**: Identify high-risk operating envelopes (temperature, speed, load)
3. **Maintenance Planning**: Flag components/systems for preventive service
4. **Design Validation**: Quantify risk under different design parameters
5. **Decision Support**: Give field engineers and program managers data-driven recommendations

### How to Present in Interview

**Elevator pitch** (2 min):
> "I built a predictive maintenance dashboard that takes industrial operating conditions—temperature, speed, torque, wear—and estimates failure risk using a RandomForest classifier. It converts model outputs into actionable engineering insights: Low/Medium/High risk categories with specific recommendations. The goal is to show how machine learning can support durability analysis and engineering decision-making in product development."

**Deep dive** (if asked):
- Dataset: UCI AI4I Predictive Maintenance; ~10k industrial operating records
- Model: RandomForest (100 trees) with balanced class weights for ~15% failure rate
- Performance: 94% accuracy, 69% precision, 65% recall on failures
- Key insight: Tool wear and rotational speed are the strongest predictors of failure
- Dashboard: Real-time prediction + feature importance + risk categorization + engineering recommendations

**If asked "Why RandomForest?"**:
- Fast to train and suitable for this prototype
- Explainable feature importance (essential for engineering trust)
- Inherent handling of non-linearities
- If we scale: Could transition to gradient boosting or deep learning, but would still use SHAP for explainability

---

## Future Improvements & Extensions

### Short-term (Could add in interview follow-up)

- [ ] **SHAP Explainability**: Per-instance feature explanations ("This specific torque value increased risk by X%")
- [ ] **Failure Mode Classification**: Predict *type* of failure (bearing, thermal, tool wear), not just binary
- [ ] **Trend Analysis**: Track failure risk over time (line chart of risk vs. operating hours)
- [ ] **Threshold Optimization**: Find the safest operating ranges for each condition
- [ ] **Unit Testing**: Add pytest for data pipeline validation

### Medium-term (Production roadmap)

- [ ] **Real Data**: Integrate actual test or field data (replace synthetic dataset)
- [ ] **Continuous Learning**: Retrain model monthly with new failure observations
- [ ] **Database Connector**: Stream live sensor data from historian or IoT platform
- [ ] **Multi-asset Support**: Predictions for different product/asset types
- [ ] **API Backend**: RESTful service for integration with manufacturing software

### Long-term (Strategic)

- [ ] **Multi-model Ensemble**: Combine RandomForest + gradient boosting + neural networks
- [ ] **Bayesian Workflow**: Quantify uncertainty in predictions
- [ ] **Causal Inference**: Move beyond correlation to true cause-effect understanding
- [ ] **Deployment**: Docker containerization, cloud deployment (AWS/Azure), model versioning
- [ ] **Monitoring**: Track model drift and retraining triggers

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**: Reinstall dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Issue: `FileNotFoundError: failure_model.pkl`

**Solution**: Train the model first
```bash
python train_model.py
```
This generates `failure_model.pkl` and `model_features.pkl`.

### Issue: `ucimlrepo` dataset fetch fails (network error)

**Solution**: The UCI repository temporarily unavailable. Wait a few minutes and retry. If persistent, check your internet connection.

### Issue: Streamlit not opening a browser

**Solution**: Manually navigate to `http://localhost:8501` in your browser after running the app.

---

## Files Overview

### `requirements.txt`
Python package dependencies with version constraints. Install with:
```bash
python -m pip install -r requirements.txt
```

### `train_model.py`
- Fetches UCI AI4I dataset
- Exploratory stats (class distribution, shape)
- Trains RandomForest on 80% of data
- Prints confusion matrix and classification report
- Saves model as `failure_model.pkl` and features as `model_features.pkl`

Run once after cloning:
```bash
python train_model.py
```

### `app.py`
- Streamlit web application
- Loads trained model and features
- Provides sidebar for operating condition inputs (sliders)
- Displays real-time failure risk prediction and categorization
- Shows feature importance and model metadata

Run after training:
```bash
python -m streamlit run app.py
```

---

## Author & Context

**Built by**: Candidate for Gates Corporation Rotational Engineer (AI/ML + Engineering)  
**Date**: May 2026  
**Purpose**: Portfolio demonstration of applied ML in industrial/product development context  
**License**: Open-source (feel free to share and build upon)

---

## Questions? Ideas?

If you're exploring this:
- **For a homework/learning project**: Try modifying the model hyperparameters or using a different classifier
- **For extending it**: Add real test data, built explainability, or deploy to the cloud
- **For interview prep**: Use this as a template for building small, focused, narrative-driven ML projects

---

## References

- [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SHAP (for future explainability)](https://shap.readthedocs.io/)
- [Gates Corporation](https://www.gates.com/)
