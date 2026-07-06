"""
train_baseline.py — Train and evaluate the baseline Logistic Regression model.

Pipeline: TF-IDF(1,2-gram, 20k features) → LogisticRegression(balanced).
Fits on train split only, evaluates on val split.
Saves model to models/v0.1_baseline.joblib and metrics to
models/v0.1_baseline_metrics.json.

Does NOT evaluate on test.csv — that is reserved for Phase 2.
"""

import sys
import os
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.tfidf import build_tfidf_vectorizer

# ---------------------------------------------------------------------------
# Project-wide reproducibility seed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42


def train_baseline(
    train_path: str = "data/splits/train.csv",
    val_path: str = "data/splits/val.csv",
    model_dir: str = "models",
) -> dict:
    """Train baseline model and evaluate on validation set.

    Returns
    -------
    dict
        Computed evaluation metrics.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading training data from {train_path} ...")
    df_train = pd.read_csv(train_path)
    print(f"Loading validation data from {val_path} ...")
    df_val = pd.read_csv(val_path)

    print(f"  Train rows: {len(df_train):,}")
    print(f"  Val rows:   {len(df_val):,}")

    # ------------------------------------------------------------------
    # 2. Edge case: verify no empty clean_text in splits
    #    If any exist, Step 2's drop logic failed — raise an explicit
    #    error rather than silently producing meaningless all-zero
    #    feature vectors from TfidfVectorizer.
    # ------------------------------------------------------------------
    train_empty = (df_train["clean_text"].fillna("").str.strip() == "").sum()
    val_empty = (df_val["clean_text"].fillna("").str.strip() == "").sum()

    if train_empty > 0 or val_empty > 0:
        raise ValueError(
            f"Empty clean_text found after split! "
            f"Train empties: {train_empty}, Val empties: {val_empty}. "
            f"This indicates run_preprocessing.py's drop logic failed. "
            f"Fix Step 2 before continuing."
        )
    print("  ✅ No empty clean_text in train/val splits.")

    X_train = df_train["clean_text"]
    y_train = df_train["label"]
    X_val = df_val["clean_text"]
    y_val = df_val["label"]

    # ------------------------------------------------------------------
    # 3. Build pipeline
    #    TF-IDF vectorizer is fit ONLY on training data (via Pipeline.fit).
    #    class_weight='balanced' protects against class imbalance even if
    #    the current dataset is near-balanced — zero cost, future-proof.
    # ------------------------------------------------------------------
    vectorizer = build_tfidf_vectorizer()
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("\nTraining pipeline (TF-IDF + LogisticRegression) ...")
    pipeline.fit(X_train, y_train)
    print("  ✅ Training complete.")

    # ------------------------------------------------------------------
    # 5. Predict on validation set
    # ------------------------------------------------------------------
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]  # probability of class 1

    # ------------------------------------------------------------------
    # 6. Degenerate model check
    #    If the model predicts only one class, flag it as a failure.
    # ------------------------------------------------------------------
    unique_preds = set(y_pred)
    if len(unique_preds) == 1:
        raise RuntimeError(
            f"DEGENERATE MODEL: Only class {unique_preds.pop()} predicted "
            f"across the entire validation set ({len(y_val)} rows). "
            f"The model is not learning to discriminate between classes. "
            f"Investigate feature extraction and class balance."
        )
    print(f"  ✅ Both classes predicted: {sorted(unique_preds)}")

    # ------------------------------------------------------------------
    # 7. Compute metrics
    # ------------------------------------------------------------------
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="macro")
    rec = recall_score(y_val, y_pred, average="macro")
    f1 = f1_score(y_val, y_pred, average="macro")
    roc_auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=["Real (0)", "Fake (1)"])

    print(f"\n{'='*60}")
    print(f"VALIDATION SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f} (macro)")
    print(f"  Recall:      {rec:.4f} (macro)")
    print(f"  F1 Score:    {f1:.4f} (macro)")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {'':>15} Predicted 0  Predicted 1")
    print(f"  {'Actual 0':>15}    {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"  {'Actual 1':>15}    {cm[1][0]:>6}       {cm[1][1]:>6}")
    print(f"\nClassification Report:")
    print(report)

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "v0.1_baseline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"  Saved model → {model_path}")

    # ------------------------------------------------------------------
    # 9. Save metrics JSON
    # ------------------------------------------------------------------
    tfidf_params = pipeline.named_steps["tfidf"].get_params()
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "accuracy": round(acc, 6),
        "precision_macro": round(prec, 6),
        "recall_macro": round(rec, 6),
        "f1_macro": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
        "tfidf_params": {
            "ngram_range": list(tfidf_params["ngram_range"]),
            "max_features": tfidf_params["max_features"],
            "min_df": tfidf_params["min_df"],
            "max_df": tfidf_params["max_df"],
        },
        "classifier": "LogisticRegression",
        "classifier_params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    }

    metrics_path = os.path.join(model_dir, "v0.1_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics → {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_baseline()
