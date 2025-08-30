import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_eda(df, out_dir="outputs"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Save summary statistics
    df.describe().to_csv(out_dir / "data_describe.csv")

    # Region-based numeric means
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.groupby("region")[numeric_cols].mean().to_csv(out_dir / "region_means.csv")

    # Histograms
    for col in ["avg_weekly_earnings", "trips_per_week", "mean_rating",
                "cancellation_rate", "on_time_pct", "tenure_months"]:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png")
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_heatmap.png")
    plt.close()

    print(f"EDA results saved in {out_dir}")
