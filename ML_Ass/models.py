# train_models.py
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost
from xgboost import XGBClassifier

# ----------------------------
# Config
# ----------------------------
RANDOM_STATE = 42
from sklearn.datasets import load_breast_cancer

# Use Breast Cancer Wisconsin (Diagnostic) dataset instead of Wine
DATA_URL = None
LOCAL_DATA = Path("data/breast_cancer_wisconsin.csv")
MODELS_DIR = Path("models")
FIGS_DIR = MODELS_DIR / "figs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
def load_breast_cancer_df():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset via sklearn
    and return a DataFrame, features X and target y.
    The output DataFrame has feature columns and a `target` column (0=malignant,1=benign).
    """
    if LOCAL_DATA.exists():
        df = pd.read_csv(LOCAL_DATA)
    else:
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        # cache locally for reproducibility
        os.makedirs(LOCAL_DATA.parent, exist_ok=True)
        df.to_csv(LOCAL_DATA, index=False)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    return df, X, y

# ----------------------------
# Build Models
# ----------------------------
def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    return models

# ----------------------------
# Metrics helpers
# ----------------------------
def multiclass_auc(y_true, proba, classes):
    """
    Macro One-vs-Rest AUC for multiclass.
    y_true: array-like (n_samples,)
    proba: array-like (n_samples, n_classes) -> class probabilities
    classes: list of class labels in sorted order aligned with proba columns
    """
    # binarize y
    lb = LabelBinarizer()
    lb.fit(classes)
    y_bin = lb.transform(y_true)
    # Handle case of single positive in any class (rare), guard against errors
    try:
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan
    return auc


def compute_and_plot_cm(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(np.unique(y_true)),
                yticklabels=sorted(np.unique(y_true)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    df, X, y = load_breast_cancer_df()

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Save a ready-to-upload test CSV for Streamlit app
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df.to_csv(MODELS_DIR / 'test_holdout.csv', index=False)

    models = build_models()
    rows = []

    classes_sorted = sorted(np.unique(y_train))

    for name, model in models.items():
        # Fit (handle XGBoost which expects zero-based class indices)
        is_xgb = False
        if isinstance(model, XGBClassifier) or 'xgboost' in name.lower():
            classes_sorted = sorted(np.unique(y_train))
            label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
            y_train_enc = np.array([label_to_idx[yv] for yv in y_train])
            model.fit(X_train, y_train_enc)
            is_xgb = True
        else:
            model.fit(X_train, y_train)

        # Save model
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_path)

        # Predict / Proba
        y_pred = model.predict(X_test)
        if is_xgb:
            y_pred = np.array([classes_sorted[int(p)] for p in y_pred])
        # Probability handling
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_test)
            if df_scores.ndim == 1:
                df_scores = np.vstack([-df_scores, df_scores]).T
            exp = np.exp(df_scores - np.max(df_scores, axis=1, keepdims=True))
            proba = exp / np.sum(exp, axis=1, keepdims=True)
        else:
            proba = np.zeros((len(y_pred), len(classes_sorted)))
            class_to_index = {c: i for i, c in enumerate(classes_sorted)}
            for i, p in enumerate(y_pred):
                proba[i, class_to_index[p]] = 1.0

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = multiclass_auc(y_test, proba, classes_sorted)

        # Plot and save CM
        fig_path = FIGS_DIR / f"{safe_name}_cm.png"
        compute_and_plot_cm(y_test, y_pred, f"Confusion Matrix — {name}", fig_path)

        # Classification report (saved per model for reference)
        report = classification_report(y_test, y_pred, digits=4)
        with open(MODELS_DIR / f"{safe_name}_report.txt", "w") as f:
            f.write(report)

        rows.append({
            "ML Model Name": name,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc
        })

        print(f"\n{name}\n{'-'*len(name)}\n{report}")

    # Save metrics table
    metrics_df = pd.DataFrame(rows)
    metrics_df.sort_values("F1", ascending=False, inplace=True)
    metrics_df.to_csv(MODELS_DIR / "metrics_summary.csv", index=False)

    print("\nSaved models to ./models and metrics to ./models/metrics_summary.csv")
    print("Test holdout CSV saved to ./models/test_holdout.csv (upload this to the Streamlit app)")

if __name__ == "__main__":
    main()