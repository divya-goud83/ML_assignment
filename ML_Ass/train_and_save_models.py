# train_and_save_models.py

# Purpose: Train 6 classifiers on UCI Bank Marketing dataset and save fitted pipelines + metrics.

import os
import io
import json
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

import joblib

# Optional: xgboost (make sure it's in requirements.txt)
from xgboost import XGBClassifier

# -------------------------------
# 0) Repro & output structure
# -------------------------------
RANDOM_STATE = 42
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = OUT_DIR / "metrics_summary.csv"
DETAILED_REPORTS_DIR = OUT_DIR / "reports"
DETAILED_REPORTS_DIR.mkdir(exist_ok=True)

# -------------------------------
# 1) Download & load data (UCI)
# -------------------------------
# UCI "bank-additional-full.csv" is in a zip: 00222/bank-additional.zip
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
CSV_IN_ZIP = "bank-additional/bank-additional-full.csv"  # ; separated

def load_uci_bank_marketing():
    try:
        print("Downloading UCI bank-additional.zip ...")
        with urllib.request.urlopen(UCI_ZIP_URL) as resp:
            zip_bytes = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        with zf.open(CSV_IN_ZIP) as f:
            df = pd.read_csv(f, sep=';')
        print("Loaded UCI dataset from web.")
        return df
    except Exception as e:
        # Fallback: try local file if present
        local_candidate = Path("bank-additional-full.csv")
        if local_candidate.exists():
            df = pd.read_csv(local_candidate, sep=';')
            print("Loaded local bank-additional-full.csv")
            return df
        raise RuntimeError(f"Failed to download/load dataset: {e}")

df = load_uci_bank_marketing()

# -------------------------------
# 2) Basic cleaning
# -------------------------------
# Drop known leakage column (duration)
if "duration" in df.columns:
    df = df.drop(columns=["duration"])

# Target column 'y' with values 'yes'/'no' -> map to 1/0
if "y" not in df.columns:
    raise ValueError("Expected target column 'y' not found.")
df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)

# Identify features
target = "y"
feature_cols = [c for c in df.columns if c != target]
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

# -------------------------------
# 3) Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df[target],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df[target]
)

# -------------------------------
# 4) Preprocessing
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_sparse = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # sparse OK for many models
])

categorical_transformer_dense = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Standard preprocessor (sparse for efficiency)
preprocessor_sparse = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_sparse, cat_cols),
    ],
    remainder="drop"
)

# Dense preprocessor (for GaussianNB)
preprocessor_dense = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer_dense, cat_cols),
    ],
    remainder="drop"
)

# -------------------------------
# 5) Define models (6 required)
# -------------------------------
models = {
    "Logistic Regression": Pipeline(steps=[
        ("pre", preprocessor_sparse),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE
        ))
    ]),
    "Decision Tree": Pipeline(steps=[
        ("pre", preprocessor_sparse),
        ("clf", DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ]),
    "kNN": Pipeline(steps=[
        ("pre", preprocessor_sparse),
        ("clf", KNeighborsClassifier(n_neighbors=15))
    ]),
    "Naive Bayes (Gaussian)": Pipeline(steps=[
        ("pre", preprocessor_dense),
        ("clf", GaussianNB())
    ]),
    "Random Forest": Pipeline(steps=[
        ("pre", preprocessor_sparse),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE
        ))
    ]),
    "XGBoost": Pipeline(steps=[
        ("pre", preprocessor_sparse),
        ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1, reg_lambda=1.0
        ))
    ]),
}

# -------------------------------
# 6) Train, evaluate, persist
# -------------------------------
def get_proba(model, X):
    # use predict_proba if available, else decision_function → scale to 0-1
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        scores = model.decision_function(X).reshape(-1, 1)
        return MinMaxScaler().fit_transform(scores).ravel()
    else:
        return model.predict(X).astype(float)

rows = []
for name, pipe in models.items():
    print(f"\nTraining: {name}")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = get_proba(pipe, X_test)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = np.nan
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Persist model pipeline
    model_path = OUT_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
    joblib.dump(pipe, model_path)

    # Save per-model text report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["no", "yes"], zero_division=0)
    reports_dir = OUT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / f"{name}_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {name}\n\n")
        f.write("Confusion Matrix [rows=true, cols=pred]:\n")
        import pandas as pd
        f.write(pd.DataFrame(cm, index=["no","yes"], columns=["no","yes"]).to_string())
        f.write("\n\nClassification Report:\n")
        f.write(report)

    rows.append({
        "ML Model Name": name,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Saved Model": str(model_path)
    })

# Metrics summary table
metrics_df = pd.DataFrame(rows)
metrics_df.sort_values(by=["F1", "AUC"], ascending=False, inplace=True)
metrics_df.to_csv(METRICS_CSV, index=False)
print("\n=== Metrics Summary ===")
print(metrics_df)

# Save a small schema file for the app
schema = {
    "feature_columns": [c for c in df.columns if c != target],
    "numeric_columns": [c for c in df.columns if c != target and df[c].dtype != 'object'],
    "categorical_columns": [c for c in df.columns if c != target and df[c].dtype == 'object'],
    "target": target,
    "dropped_columns": ["duration"],
}
with open(OUT_DIR / "schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
