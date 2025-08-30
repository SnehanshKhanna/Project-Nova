import pandas as pd
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path

def evaluate_fairness(df, preds, sensitive_attr="gender", out_dir="outputs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = df.copy()
    df["pred"] = preds

    groups = df[sensitive_attr].unique()
    fairness_summary = {}

    for g in groups:
        sub = df[df[sensitive_attr] == g]
        tn, fp, fn, tp = confusion_matrix(sub["creditworthy"], sub["pred"]).ravel()
        tpr = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)
        fairness_summary[g] = {"TPR": tpr, "FPR": fpr}

    with open(out_dir / "fairness_summary.json", "w") as f:
        json.dump(fairness_summary, f, indent=4)

    print("Fairness results saved.")
    return fairness_summary
