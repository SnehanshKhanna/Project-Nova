# 🌟 Project Nova  

**Synthetic Data Generation, EDA, and Machine Learning Models for Performance Prediction**  

---

## 📌 Project Overview  
This project shows a complete **data science pipeline** from start to finish.  
Since we don’t have real-world data, we generate **synthetic (fake but realistic) data** about drivers, then:  

1. **Analyze it** with exploratory data analysis (EDA)  
2. **Train machine learning models** to predict performance  
3. **Check fairness** of the models across groups (like gender or region)  

The project is modular, so each part is handled by a separate file, and everything can also be run end-to-end with a single script.  

---

## 🗂️ Project Structure  

```text
Nova/
│── data_gen.py      # Generates synthetic driver data
│── eda.py           # Runs exploratory data analysis and saves graphs/stats
│── train.py         # Trains ML models and evaluates performance
│── fairness.py      # Checks fairness/bias of the models
│── utils.py         # Helper functions (keeps code clean)
│── main.py          # Orchestrates the full pipeline (run this to execute all steps)
│── outputs/         # Stores generated datasets, graphs, and trained models
│── requirements.txt # List of required Python libraries
│── README.md        # Project description and usage guide



---

## ⚙️ How It Works  

1. **`data_gen.py`** → Creates synthetic driver dataset (features like income, trips, ratings, etc.).  
2. **`eda.py`** → Runs EDA and generates plots (distributions, correlations, etc.).  
3. **`train.py`** → Trains Logistic Regression and Random Forest models, outputs metrics (accuracy, AUC, precision, recall).  
4. **`fairness.py`** → Checks whether the models are fair across groups (e.g., gender, region).  
5. **`main.py`** → Runs everything in sequence:  
   - Generate data  
   - Perform EDA  
   - Train models  
   - Run fairness checks  

---

## 📊 Example Outputs  

- **EDA Graphs**: Histograms, correlation heatmaps  
- **Model Performance**: Accuracy, AUC, precision/recall  
- **Fairness Analysis**: Metrics split by groups  

All results are saved in the `outputs/` folder.  

---

## 🚀 How to Run  

1. Clone the repo or download the folder.  
2. Install dependencies:  

```bash
pip install -r requirements.txt
Run the full pipeline:

bash
Copy code
python main.py
Check the outputs/ folder for graphs, CSV files, and trained models.

🛠️ Requirements
Key libraries used in the project:

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

(Full list is in requirements.txt)

✨ In Short
Goal: Demonstrate an end-to-end data science workflow

Data: Synthetic driver dataset

Steps: Data → EDA → Machine Learning → Fairness check

Outputs: Visuals, metrics, and saved models