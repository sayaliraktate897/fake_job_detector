import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="wide")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "best_model.joblib"
METRICS_PATH = BASE / "model" / "metrics.json"

st.title("🕵️ Fake Job Posting Detector")
st.markdown("### Paste a job posting and the model predicts whether it is real or fake")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()

tab1, tab2 = st.tabs(["🔍 Check a job posting", "📊 Model performance"])

# ---------- Tab 1: prediction ----------
with tab1:
    title = st.text_input("Job title")
    text = st.text_area("Job description / requirements / company profile", height=250)

    if st.button("Check posting"):
        full_text = f"{title} {text}".lower().strip()
        if not full_text:
            st.warning("Please enter some text first.")
        else:
            prob_fake = float(model.predict_proba([full_text])[0][1])
            if prob_fake >= 0.5:
                st.error(f"⚠️ Likely FAKE (fake probability {prob_fake * 100:.1f}%)")
            else:
                st.success(f"✅ Likely REAL (fake probability {prob_fake * 100:.1f}%)")
            st.progress(min(max(prob_fake, 0.0), 1.0))
            st.caption("This is a model prediction, not a guarantee. Always verify the company yourself.")

# ---------- Tab 2: metrics ----------
with tab2:
    st.subheader("Model comparison (test set)")
    rows = []
    for name, r in metrics["results"].items():
        rows.append({"Model": name, "Accuracy": r["accuracy"], "Precision (fake)": r["precision"],
                     "Recall (fake)": r["recall"], "F1 (fake)": r["f1"], "ROC-AUC": r["roc_auc"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(f"Best model (highest F1 on fake postings): **{metrics['best_model']}**. "
            f"Train size {metrics['train_size']}, test size {metrics['test_size']}.")

    st.subheader(f"Confusion matrix: {metrics['best_model']}")
    cm = metrics["results"][metrics["best_model"]]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted real", "Predicted fake"], y=["Actually real", "Actually fake"])
    st.plotly_chart(fig, use_container_width=True)

st.caption("Developed by: Sayali Dattatray Raktate")
