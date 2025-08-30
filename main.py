import os
import joblib
import pandas as pd
from data_gen import generate_synthetic_data
from eda import run_eda
from train import train_models

def run_pipeline():
    print("✅ Generating synthetic data...")
    df = generate_synthetic_data(n_samples=5000)
    print("✅ Synthetic data generated")

    print("Running EDA...")
    run_eda(df, out_dir="outputs")
    print("✅ EDA complete")

    print("Training models...")
    results = train_models(df, out_dir="outputs")
    print("✅ Models trained")

    # Load models after saving
    log_reg_model = joblib.load("outputs/logistic_regression.joblib")
    rf_model = joblib.load("outputs/random_forest.joblib")

    print("\n📊 Final Model Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")

    # Inference step
    print("\n🔮 Running inference on a random partner profile...")
    sample = df.drop(["partner_id", "creditworthy", "is_good"], axis=1).iloc[[0]]  # first row
    sample_encoded = pd.get_dummies(sample, drop_first=True)

    # Ensure same columns as training data
    all_features = pd.get_dummies(
        df.drop(["partner_id", "creditworthy", "is_good"], axis=1), drop_first=True
    ).columns
    for col in all_features:
        if col not in sample_encoded.columns:
            sample_encoded[col] = 0
    sample_encoded = sample_encoded[all_features]

    log_pred = log_reg_model.predict(sample_encoded)[0]
    rf_pred = rf_model.predict(sample_encoded)[0]

    print(f"Logistic Regression prediction: {'Creditworthy' if log_pred == 1 else 'Not Creditworthy'}")
    print(f"Random Forest prediction: {'Creditworthy' if rf_pred == 1 else 'Not Creditworthy'}")

if __name__ == "__main__":
    run_pipeline()
