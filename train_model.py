"""
Fake Job Posting Detection - model training
Run:  python train_model.py
Needs: data/fake_job_postings.csv  (Kaggle: "Real or Fake Job Posting Prediction")
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "archive (10)" / "fake_job_postings.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]


# ---------- 1. Load and clean ----------
def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"Rows: {len(df)}")
    print("Real vs fake counts:\n", df["fraudulent"].value_counts(), "\n")

    # fill missing text and join the text columns into one column
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("")
    df["text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1).str.lower()
    df = df[df["text"].str.strip() != ""]
    return df[["text", "fraudulent"]]


# ---------- 2. Models ----------
def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000,
                                  ngram_range=(1, 2), min_df=3)),
        ("model", model),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=42),
}


def evaluate(pipe, X_test, y_test):
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    df = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["fraudulent"], test_size=0.2, stratify=df["fraudulent"], random_state=42)

    results, pipes = {}, {}
    for name, model in MODELS.items():
        print(f"Training {name} ...")
        pipe = make_pipeline(model).fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        pipes[name] = pipe
        r = results[name]
        print(f"  accuracy={r['accuracy']}  precision={r['precision']}  "
              f"recall={r['recall']}  f1={r['f1']}  roc_auc={r['roc_auc']}")

    # best model = highest F1 on the fake class (accuracy is misleading here)
    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest model: {best}")

    joblib.dump(pipes[best], MODEL_DIR / "best_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best, "results": results,
                   "train_size": len(X_train), "test_size": len(X_test)}, f, indent=2)
    print("Saved model/best_model.joblib and model/metrics.json")


if __name__ == "__main__":
    main()
