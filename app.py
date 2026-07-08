"""
Network Intrusion Detection - Streamlit Prototype
M.Sc. Applied Computer Science - SRH University Heidelberg
Author: Atharva Gajbe

Run with: streamlit run app.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
DATA_DIR = "data"

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    return model, scaler, encoders, metadata


def missing_artifacts():
    required = [
        "best_model.joblib", "scaler.joblib", "label_encoders.joblib", "model_metadata.json",
    ]
    return [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]


def encode_categorical(df, encoders, cat_cols):
    """Encode categorical columns with saved LabelEncoders. Unseen categories map to -1."""
    df = df.copy()
    unseen = {}
    for col in cat_cols:
        le = encoders[col]
        known = set(le.classes_)
        values = df[col].astype(str)
        unseen_vals = sorted(set(values) - known)
        if unseen_vals:
            unseen[col] = unseen_vals
        class_to_idx = {c: i for i, c in enumerate(le.classes_)}
        df[col] = values.map(lambda v: class_to_idx.get(v, -1))
    return df, unseen


def predict(df_raw, model, scaler, encoders, feature_cols, cat_cols):
    missing_cols = [c for c in feature_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df = df_raw[feature_cols].copy()
    df, unseen = encode_categorical(df, encoders, cat_cols)

    X = scaler.transform(df.values)
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = preds.astype(float)

    result = df_raw.copy()
    result["prediction"] = np.where(preds == 1, "Malicious", "Normal")
    result["confidence"] = np.where(preds == 1, proba, 1 - proba)
    return result, unseen


def main():
    st.title("🛡️ Network Intrusion Detection System")
    st.caption("ML-based classification of network traffic as Normal or Malicious — NSL-KDD dataset")

    missing = missing_artifacts()
    if missing:
        st.error(
            "Model artifacts not found: " + ", ".join(missing) + ". "
            "Run notebooks/01_data_preprocessing.ipynb and notebooks/02_model_training.ipynb first."
        )
        st.stop()

    model, scaler, encoders, metadata = load_artifacts()
    feature_cols = metadata["feature_columns"]
    cat_cols = metadata["categorical_columns"]

    with st.sidebar:
        st.subheader("Model Info")
        st.write(f"**Algorithm:** {metadata['best_model_name']}")
        m = metadata["metrics"]
        st.metric("Accuracy", f"{m['Accuracy']:.3f}")
        st.metric("Recall", f"{m['Recall']:.3f}")
        st.metric("F1-Score", f"{m['F1-Score']:.3f}")
        st.caption(
            f"5-fold CV F1: {metadata['cv_f1_mean']:.3f} ± {metadata['cv_f1_std']:.3f}"
        )
        st.divider()
        st.caption(
            "Trained on NSL-KDD. Recall is prioritized because a missed "
            "attack (false negative) is more costly than a false alarm."
        )

    tab_csv, tab_manual = st.tabs(["📁 Upload CSV", "✍️ Manual Input"])

    with tab_csv:
        st.write(
            "Upload a CSV with NSL-KDD style traffic records "
            f"(columns: `{', '.join(feature_cols[:5])}, ...`)."
        )
        sample_path = os.path.join(DATA_DIR, "sample_traffic.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    "Download sample CSV", f, file_name="sample_traffic.csv", mime="text/csv"
                )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df_raw = pd.read_csv(uploaded)
                if df_raw.empty:
                    st.warning("The uploaded CSV is empty.")
                else:
                    result, unseen = predict(df_raw, model, scaler, encoders, feature_cols, cat_cols)
                    if unseen:
                        st.warning(
                            "Some categorical values were not seen during training and were "
                            f"treated as unknown: {unseen}"
                        )
                    n_malicious = (result["prediction"] == "Malicious").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Records", len(result))
                    c2.metric("Normal", int((result["prediction"] == "Normal").sum()))
                    c3.metric("Malicious", int(n_malicious))

                    def highlight(row):
                        color = "background-color: #ffe3e3" if row["prediction"] == "Malicious" else ""
                        return [color] * len(row)

                    st.dataframe(result.style.apply(highlight, axis=1), use_container_width=True)
                    st.download_button(
                        "Download results as CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Could not process file: {e}")

    with tab_manual:
        st.write("Enter feature values for a single connection record.")
        with st.form("manual_form"):
            values = {}
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Categorical features**")
                for col in cat_cols:
                    options = list(encoders[col].classes_)
                    default = "tcp" if col == "protocol_type" and "tcp" in options else options[0]
                    values[col] = st.selectbox(col, options, index=options.index(default))
            with col2:
                st.markdown("**Key numeric features**")
                key_numeric = [
                    "duration", "src_bytes", "dst_bytes", "count", "srv_count",
                    "logged_in", "serror_rate", "same_srv_rate",
                ]
                for col in key_numeric:
                    values[col] = st.number_input(col, value=0.0, step=1.0)

            with st.expander("Other numeric features (defaults to 0)"):
                remaining = [c for c in feature_cols if c not in cat_cols and c not in values]
                cols = st.columns(3)
                for i, col in enumerate(remaining):
                    with cols[i % 3]:
                        values[col] = st.number_input(col, value=0.0, step=1.0, key=f"extra_{col}")

            submitted = st.form_submit_button("Predict")

        if submitted:
            row = {c: values[c] for c in feature_cols}
            df_raw = pd.DataFrame([row])
            result, unseen = predict(df_raw, model, scaler, encoders, feature_cols, cat_cols)
            pred = result.iloc[0]["prediction"]
            conf = result.iloc[0]["confidence"]

            if pred == "Malicious":
                st.error(f"🚨 Prediction: **{pred}** (confidence: {conf:.1%})")
            else:
                st.success(f"✅ Prediction: **{pred}** (confidence: {conf:.1%})")


if __name__ == "__main__":
    main()
