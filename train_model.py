"""
train_model.py

Trains a RandomForest classifier on the UCI AI4I 2020 Predictive Maintenance dataset.
Outputs:
  - failure_model.pkl: trained model
  - Confusion matrix and classification report to console
"""

import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    print("=" * 60)
    print("Predictive Failure Risk Model - Training")
    print("=" * 60)
    
    # Fetch AI4I 2020 Predictive Maintenance dataset
    print("\n[1] Fetching UCI AI4I 2020 Predictive Maintenance Dataset...")
    dataset = fetch_ucirepo(id=601)
    
    X = dataset.data.features
    y = dataset.data.targets["Machine failure"]
    
    print(f"    Dataset size: {len(X)} samples")
    
    # Keep only the engineering-style numeric features
    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
    
    X = X[features]
    print(f"    Features: {features}")
    
    # Train/test split
    print("\n[2] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    print(f"    Class distribution (train): {y_train.value_counts().to_dict()}")
    
    # Train RandomForest classifier
    print("\n[3] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("    Model training complete.")
    
    # Evaluate
    print("\n[4] Model Evaluation")
    predictions = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, predictions)
    
    print(f"    Train accuracy: {train_acc:.4f}")
    print(f"    Test accuracy: {test_acc:.4f}")
    
    print("\n    Confusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    print("\n    Classification Report:")
    print(classification_report(y_test, predictions, target_names=["No Failure", "Failure"]))
    
    # Feature importance
    print("\n[5] Feature Importance")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.to_string(index=False))
    
    # Save model
    print("\n[6] Saving Model...")
    joblib.dump(model, "failure_model.pkl")
    joblib.dump(features, "model_features.pkl")
    print("    Saved: failure_model.pkl, model_features.pkl")
    
    print("\n" + "=" * 60)
    print("Training complete. Ready to run dashboard with:")
    print("  python -m streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()