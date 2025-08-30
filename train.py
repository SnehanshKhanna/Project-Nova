import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

def train_models(df, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop(["partner_id", "creditworthy", "is_good"], axis=1)
    y = df["creditworthy"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Safeguard: check class balance in train set
    if len(y_train.unique()) < 2:
        print("⚠️ Training set had only one class, rebalancing...")
        from data_gen import generate_synthetic_data
        df_new = generate_synthetic_data(len(df))
        return train_models(df_new, out_dir)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": acc,
            "auc": auc,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

        # Save trained model
        filename = os.path.join(out_dir, f"{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(model, filename)

        # Save feature importances for Random Forest
        if name == "Random Forest":
            importances = pd.Series(model.feature_importances_, index=X.columns)
            plt.figure(figsize=(8, 6))
            importances.sort_values(ascending=False).head(10).plot(kind="barh")
            plt.title("Top 10 Feature Importances (Random Forest)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rf_feature_importances.png"))
            plt.close()

    # Save results summary
    results_df = pd.DataFrame({
        model: {"accuracy": metrics["accuracy"], "auc": metrics["auc"]}
        for model, metrics in results.items()
    }).T
    results_df.to_csv(os.path.join(out_dir, "model_results.csv"))

    return results
