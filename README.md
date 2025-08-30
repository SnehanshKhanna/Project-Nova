🚀 Project Nova — Equitable Credit Scoring

Hackathon Project | Machine Learning + Fairness in AI

📌 Overview

Many gig economy workers are considered “credit invisible” because they lack traditional financial histories (like bank loans or credit cards).
Project Nova aims to create an equitable, data-driven credit scoring engine that generates a Nova Score (300–850) using alternative data such as:

Earnings history

Trip frequency

Customer ratings

Driving/merchant behavior

The model also includes a fairness analysis to ensure that demographic factors (e.g., gender, region) do not unfairly penalize workers.

✨ Features

✅ Synthetic Dataset Generator (~10k samples of simulated Grab partners)

✅ Exploratory Data Analysis (EDA) with histograms & correlation heatmap

✅ Machine Learning Models: Logistic Regression & Random Forest

✅ Fairness Metrics: Demographic Parity & Equal Opportunity differences

✅ Bias Mitigation: Reweighting of training samples for fairer predictions

✅ Nova Score Mapping: Predicted probabilities → Score (300–850 scale)

✅ Artifacts Saved: Models, CSVs, JSON fairness summary, plots

📂 Project Structure
Project-Nova/
│── main.py                # Entry point – runs the full pipeline
│── data_gen.py            # Synthetic dataset generation
│── eda.py                 # Exploratory Data Analysis
│── train.py               # Model training & evaluation
│── fairness.py            # Fairness metrics & bias analysis
│── utils.py               # Helper utilities (Nova score mapping, etc.)
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── LICENSE                # License file (MIT recommended)
│── outputs/               # Generated graphs, models, CSVs (ignored in Git)

⚙️ Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/Project-Nova.git
cd Project-Nova

2️⃣ Create virtual environment (recommended)
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Usage

Run the full pipeline:

python main.py


Outputs will be saved to the outputs/ folder:

synthetic_partners.csv → generated dataset

hist_*.png → histograms of features

corr_heatmap.png → feature correlation matrix

region_means.csv → average values per region

predictions_sample.csv → predicted Nova scores vs true values

fairness_summary.json → fairness metrics report

Trained models (.joblib)

📊 Example Outputs
Distribution of Nova Scores

Feature Correlation Heatmap

(Note: Add these screenshots once you have them in your outputs/ folder.)

🔍 Fairness in AI

The project analyzes:

Demographic Parity Difference → measures difference in loan approval rates across groups.

Equal Opportunity Difference → compares true positive rates across groups.

➡️ Example: Ensuring female/rural workers are not unfairly penalized compared to male/urban workers with similar reliability.

🛠️ Tech Stack

Python 3.8+

Libraries:

numpy, pandas → data handling

scikit-learn → ML models & preprocessing

matplotlib, seaborn → visualizations

fairlearn (optional) → fairness analysis

joblib → model persistence

📌 Future Improvements

🔹 Advanced fairness techniques (adversarial debiasing, equalized odds post-processing)

🔹 Explainability with SHAP values for feature importance

🔹 Integration into a simple API (Flask/FastAPI) for real-time scoring

🔹 Deployment on cloud (Heroku, AWS, or Streamlit app demo)

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

🙌 Acknowledgments

Hackathon problem statement inspiration from Grab / Project Nova

Libraries: Scikit-learn
, Fairlearn
, Seaborn