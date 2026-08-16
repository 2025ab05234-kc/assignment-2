import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_classification_report,
)

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

SAVED_DIR = "model/saved"
MODEL_NAME_TO_FILE = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree Classifier": "decision_tree.joblib",
    "K-Nearest Neighbor Classifier": "knn.joblib",
    "Naive Bayes Classifier - Gaussian": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib"
}


@st.cache_data
def load_data():
    df = pd.read_csv("data/breast_cancer.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    df["target_bin"] = (df["diagnosis"] == "M").astype(int)
    return df


def _eval_model(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.35).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = None
    m = compute_binary_metrics(y_test, y_pred, y_proba)
    return {"y_pred": y_pred, "y_proba": y_proba, "metrics": m}


@st.cache_resource
def load_models(X_test, y_test):
    model_names = list(MODEL_NAME_TO_FILE.keys())

    fitted_models = {}
    for name in model_names:
        path = os.path.join(SAVED_DIR, MODEL_NAME_TO_FILE[name])
        model = joblib.load(path)
        ev = _eval_model(model, X_test, y_test)
        fitted_models[name] = {"model": model, **ev}
    return fitted_models


def main():
    st.title("Breast Cancer Wisconsin — Diagnosis Predictor")

    df = load_data()
    X = df.drop(columns=["diagnosis", "target_bin"])
    y = df["target_bin"]
    feature_names = list(X.columns)

    _, X_test_default, _, y_test_default = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    uploaded_test = st.sidebar.file_uploader(
        "Upload test data (CSV)", type=["csv"], key="test_upload"
    )
    if uploaded_test is not None:
        try:
            test_df = pd.read_csv(uploaded_test)
            test_df = test_df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
            test_df["diagnosis"] = test_df["diagnosis"].astype(str).str.strip()
            test_df["target_bin"] = (test_df["diagnosis"] == "M").astype(int)
            X_test = test_df.drop(columns=["diagnosis", "target_bin"])
            y_test = test_df["target_bin"]
            missing = set(feature_names) - set(X_test.columns)
            if missing:
                st.sidebar.error(f"Missing columns: {missing}")
                X_test, y_test = X_test_default, y_test_default
            else:
                X_test = X_test[feature_names]
                st.sidebar.success(f"Using uploaded test data: {len(X_test)} samples")
        except Exception as e:
            st.sidebar.error(str(e))
            X_test, y_test = X_test_default, y_test_default
    else:
        X_test, y_test = X_test_default, y_test_default

    fitted_models = load_models(X_test, y_test)

    selected_model = st.sidebar.selectbox(
        "Select model", list(fitted_models.keys()), key="model_select"
    )

    data = fitted_models[selected_model]
    m = data["metrics"]

    st.subheader("Evaluation metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, (name, val) in zip(
        [c1, c2, c3, c4, c5, c6],
        [
            ("Accuracy", m["accuracy"]),
            ("AUC", m["auc"]),
            ("Precision", m["precision"]),
            ("Recall", m["recall"]),
            ("F1 Score", m["f1"]),
            ("MCC", m["mcc"]),
        ],
    ):
        with col:
            st.metric(name, f"{val:.4f}")

    st.subheader("Confusion matrix & classification report")
    col_cm, col_report = st.columns(2)
    with col_cm:
        cm = compute_confusion_matrix(y_test, data["y_pred"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col_report:
        report = compute_classification_report(y_test, data["y_pred"])
        st.text(report)


if __name__ == "__main__":
    main()
