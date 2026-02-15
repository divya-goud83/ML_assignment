# ML_Ass: Doctor Visits Prediction App

This project is a machine learning application for predicting doctor visits using various classification models. It includes model training, evaluation, and a Streamlit web app for interactive predictions.

## Project Structure

- `streamlit_app.py` — Streamlit web application for predictions
- `train_and_save_models.py` — Script to train and save ML models
- `models.py` — Model utility functions
- `NPHA-doctor-visits.csv` — Dataset used for training
- `requirements.txt` — Python dependencies
- `models/` — Saved models and reports
    - `Decision_Tree.joblib`, `kNN.joblib`, `Logistic_Regression.joblib`, `Naive_Bayes_Gaussian.joblib`, `Random_Forest.joblib`, `XGBoost.joblib` — Trained model files
    - `metrics_summary.csv` — Model performance summary
    - `schema.json` — Data schema
    - `reports/` — Model evaluation reports

## Setup Instructions

1. **Clone the repository** and navigate to the project directory.
2. **Create a virtual environment** (optional but recommended):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Train models (if needed):**
   ```powershell
   python train_and_save_models.py
   ```
5. **Run the Streamlit app:**
   ```powershell
   & ".venv\Scripts\streamlit.exe" run streamlit_app.py
   ```

## Usage
- Open the Streamlit app in your browser (usually at http://localhost:8501).
- Upload or input data as required and view predictions and model performance.

## Requirements
- Python 3.12+
- See `requirements.txt` for all dependencies

## Authors
- Your Name Here

## License
- Specify your license here (e.g., MIT)
