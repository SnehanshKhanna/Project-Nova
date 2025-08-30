import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=5000, random_state=42):
    rng = np.random.RandomState(random_state)

    # Features
    avg_weekly_earnings = np.clip(rng.normal(loc=250, scale=100, size=n_samples), 20, 2000)
    trips_per_week = np.clip((avg_weekly_earnings / 10 + rng.normal(0, 5, n_samples)).astype(int), 0, None)
    mean_rating = np.clip(rng.normal(loc=4.6, scale=0.4, size=n_samples), 1.0, 5.0)
    cancellation_rate = np.clip(rng.beta(1.5, 20, size=n_samples), 0.0, 1.0)
    on_time_pct = np.clip(rng.beta(8, 2, size=n_samples), 0.0, 1.0)
    tenure_months = rng.randint(1, 60, size=n_samples)

    genders = rng.choice(["male", "female", "nonbinary"], size=n_samples, p=[0.7, 0.28, 0.02])
    regions = rng.choice(["urban", "suburban", "rural"], size=n_samples, p=[0.6, 0.25, 0.15])

    # Latent score
    score = (
        (avg_weekly_earnings / 100)
        + (trips_per_week / 10)
        + mean_rating * 2
        - cancellation_rate * 5
        + on_time_pct * 3
        + (tenure_months / 12)
    )

    prob_good = 1 / (1 + np.exp(-0.5 * (score - np.median(score))))
    nova_score = np.round(300 + (prob_good * (850 - 300))).astype(int)

    df = pd.DataFrame({
        "partner_id": np.arange(1, n_samples + 1),
        "avg_weekly_earnings": avg_weekly_earnings,
        "trips_per_week": trips_per_week,
        "mean_rating": mean_rating,
        "cancellation_rate": cancellation_rate,
        "on_time_pct": on_time_pct,
        "tenure_months": tenure_months,
        "gender": genders,
        "region": regions,
        "nova_score": nova_score,
        "prob_good": prob_good
    })

    # Labels
    df["creditworthy"] = (prob_good > np.median(prob_good)).astype(int)
    df["is_good"] = df["creditworthy"]

    # Ensure balanced classes
    class_0 = df[df["creditworthy"] == 0]
    class_1 = df[df["creditworthy"] == 1]
    min_len = min(len(class_0), len(class_1))

    df_balanced = pd.concat([
        class_0.sample(min_len, random_state=random_state),
        class_1.sample(min_len, random_state=random_state)
    ])

    return df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
