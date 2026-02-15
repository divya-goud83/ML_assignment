# streamlit_app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="UCI Bank Marketing - ML Classifiers", layout="wide")

MODELS_DIR = Path("models")

@st.cache_resource
def list_models():
    return sorted([p for p in MODELS_DIR.glob("*.joblib")])

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_resource
def load_schema():
    schema_path = MODELS_DIR / "schema.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def compute_metrics(y_true, y_pred, y_proba=None):
    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            results["AUC"] = roc_auc_score(y_true, y_proba)
        except Exception:
            results["AUC"] = np.nan
    else:
        results["AUC"] = np.nan
    return results

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

st.title("Bank Marketing: Model Evaluation & Inference")
st.caption("Dataset: UCI Bank Marketing (bank-additional-full.csv). Target `y` mapped to {no:0, yes:1}. `duration` excluded to avoid leakage.")

schema = load_schema()
available_models = list_models()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1) Choose a model")
    if not available_models:
        st.error("No saved models found in ./models. Please run train_and_save_models.py first.")
        st.stop()

    model_choice = st.selectbox("Saved models", options=[p.name for p in available_models])
    chosen_model_path = MODELS_DIR / model_choice
    model = load_model(chosen_model_path)
    st.success(f"Loaded: {model_choice}")

    st.markdown("**Expected columns**:")
    if schema:
        st.code(", ".join(schema.get("feature_columns", [])), language="text")

with col_right:
    st.subheader("2) Upload CSV")
    uploaded = st.file_uploader(
        "Upload test data CSV (same schema as training; delimiter ';' or ',')",
        type=["csv"], accept_multiple_files=False
    )

    if uploaded is not None:
        # Try ; first, then ,
        try:
            df = pd.read_csv(uploaded, sep=';')
            if df.shape[1] == 1:  # probably comma-separated
                uploaded.seek(0)
                df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)

        st.write("Preview:", df.head())

        # If target present, map yes/no -> 1/0
        y_true = None
        if "y" in df.columns:
            mapped = df["y"].map({"yes": 1, "no": 0})
            if mapped.isna().sum() == 0:
                y_true = mapped.astype(int).values
            df = df.drop(columns=["y"])

        # Drop leakage column if present
        if "duration" in df.columns:
            df = df.drop(columns=["duration"])

        # Ensure only expected columns (if schema known)
        if schema:
            expected = schema.get("feature_columns", [])
            missing = [c for c in expected if c not in df.columns]
            extra = [c for c in df.columns if c not in expected]
            if missing:
                st.warning(f"Missing expected columns: {missing}")
            if extra:
                st.info(f"Ignoring unexpected columns: {extra}")
                df = df[[c for c in df.columns if c in expected]]

        # Do predictions
        with st.spinner("Scoring..."):
            y_pred = model.predict(df)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(df)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(df)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
                y_proba = scores
            else:
                y_proba = None

        st.subheader("3) Results")
        if y_true is not None:
            metrics = compute_metrics(y_true, y_pred, y_proba)
            st.write(pd.DataFrame([metrics]))
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm)

            st.markdown("**Classification report**")
            report = classification_report(y_true, y_pred, target_names=["no", "yes"], zero_division=0, output_dict=False)
            st.code(report, language="text")
        else:
            st.info("No `y` column found → Displaying predictions only.")
            output = df.copy()
            output["prediction"] = y_pred
            if y_proba is not None:
                output["prob_yes"] = y_proba
            st.dataframe(output.head(50))
            st.download_button(
                "Download predictions (CSV)",
                output.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Tip: Use your hold-out test split with a `y` column to view metrics & confusion matrix.")
