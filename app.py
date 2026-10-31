import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

st.set_page_config(
    page_title="Diabetes Detection AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Green medical theme inspired by the requested reference palette.
st.markdown("""
<style>
:root { --forest:#183a2d; --green:#34764d; --mint:#eaf4ec; }
.stApp { background: linear-gradient(135deg, #f7faf7 0%, #edf5ef 100%); }
[data-testid="stSidebar"] { background: #183a2d; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
.hero {
  background: linear-gradient(115deg, #183a2d, #34764d);
  padding: 1.7rem 2rem; border-radius: 18px; color: white;
  margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(24,58,45,.15);
}
.hero h1 { color:white; margin:0; font-size:2.25rem; }
.hero p { color:#e4f1e7; margin:.5rem 0 0 0; font-size:1.05rem; }
.metric-card {
  background:white; border:1px solid #dce9df; border-radius:14px;
  padding:1rem 1.1rem; box-shadow:0 3px 12px rgba(20,60,35,.05);
}
div.stButton > button { background:#34764d; color:white; border-radius:10px; border:0; }
div.stButton > button:hover { background:#183a2d; color:white; border:0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🩺 Diabetes Detection AI</h1>
  <p>Machine Learning • Data Visualization • Preprocessing • Prediction</p>
</div>
""", unsafe_allow_html=True)

DEFAULT_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def clean_target(series):
    s = series.copy()
    s = s.dropna()
    # Standardize common Pima labels and numeric targets.
    if s.dtype == object:
        mapped = s.astype(str).str.strip().str.lower().map({
            "tested_positive": 1, "positive": 1, "diabetic": 1, "yes": 1, "true": 1,
            "tested_negative": 0, "negative": 0, "non-diabetic": 0, "no": 0, "false": 0,
        })
        if mapped.notna().all():
            return mapped.astype(int), None
    vals = list(pd.unique(s))
    if len(vals) == 2:
        try:
            nums = pd.to_numeric(pd.Series(vals))
            if set(nums.tolist()) <= {0, 1}:
                return pd.to_numeric(series, errors="coerce"), None
        except Exception:
            pass
        enc = LabelEncoder()
        encoded = pd.Series(enc.fit_transform(series.astype(str)), index=series.index)
        return encoded, enc
    return pd.to_numeric(series, errors="coerce"), None

def fit_model(df, target, model_name):
    work = df.copy()
    y, encoder = clean_target(work[target])
    X = work.drop(columns=[target])
    # Keep numeric columns for a straightforward medical-tabular workflow.
    X = X.select_dtypes(include=[np.number]).copy()
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if X.empty or y.nunique() < 2:
        raise ValueError("The selected target must contain two classes and the dataset must have numeric feature columns.")
    # Treat common medically impossible zeros as missing for Pima columns.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                "plas", "pres", "skin", "insu", "mass"]:
        if col in X.columns:
            X.loc[X[col] == 0, col] = np.nan

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    classifier = (RandomForestClassifier(n_estimators=250, random_state=42,
                                         class_weight="balanced")
                  if model_name == "Random Forest"
                  else LogisticRegression(max_iter=2000, class_weight="balanced"))
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
    }
    try:
        prob = model.predict_proba(X_test)[:, 1]
        metrics["ROC-AUC"] = roc_auc_score(y_test, prob)
    except Exception:
        prob = None
    return model, X.columns.tolist(), X_train, X_test, y_train, y_test, pred, metrics, encoder

if "model_state" not in st.session_state:
    st.session_state.model_state = None

with st.sidebar:
    st.header("⚙️ Project Setup")
    st.caption("Upload a diabetes dataset to train a model, or load a saved model.")
    uploaded_csv = st.file_uploader("Upload CSV dataset", type=["csv"])
    target_default = "Outcome"
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    st.divider()
    st.subheader("Project stages")
    st.markdown("**1.** Visualization\n\n**2.** Preprocessing\n\n**3.** Modeling\n\n**4.** Prediction")
    st.caption("Educational prototype only — not a medical diagnosis.")

df = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.session_state["uploaded_df"] = df
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
elif "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
else:
    # Common local dataset locations, if supplied with the project.
    for candidate in ["diabetes.csv", "data/diabetes.csv", "dataset/diabetes.csv"]:
        if Path(candidate).exists():
            try:
                df = pd.read_csv(candidate)
                break
            except Exception:
                pass

if df is None:
    st.info("**Start here:** upload your `diabetes.csv` dataset from the sidebar. "
            "The app will let you explore the data, train a model, review evaluation metrics, and test a sample patient.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h3>📊 Visualization</h3><p>Distributions, correlations, and class balance.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>🧹 Preprocessing</h3><p>Missing-value handling, numeric features, and scaling.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>🤖 Modeling</h3><p>Train and evaluate a classification model.</p></div>', unsafe_allow_html=True)
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

st.success(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
tabs = st.tabs(["🏠 Overview", "📊 Visualization", "🧹 Preprocessing", "🤖 Modeling", "🩺 Predict"])

with tabs[0]:
    st.subheader("Project Overview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", str(df.shape[1]))
    target_guess = "Outcome" if "Outcome" in df.columns else ("outcome" if "outcome" in df.columns else df.columns[-1])
    c.metric("Suggested target", target_guess)
    st.markdown("### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True)
    with st.expander("Show summary statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with st.expander("Show missing values"):
        missing = df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)

with tabs[1]:
    st.subheader("Exploratory Data Analysis (EDA)")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.warning("No numeric columns were found. Upload a dataset with numeric medical features.")
    else:
        left, right = st.columns(2)
        with left:
            feature = st.selectbox("Feature distribution", numeric_df.columns, key="hist_feature")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(numeric_df[feature].dropna(), kde=True, ax=ax, color="#34764d")
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**Correlation heatmap**")
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(numeric_df.corr(), cmap="Greens", annot=False, ax=ax, linewidths=.2)
            ax.set_title("Feature Correlations")
            st.pyplot(fig)
            plt.close(fig)
        if len(numeric_df.columns) >= 2:
            xcol = st.selectbox("X axis", numeric_df.columns, index=0, key="scatter_x")
            ycol = st.selectbox("Y axis", numeric_df.columns, index=min(1, len(numeric_df.columns)-1), key="scatter_y")
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.scatter(numeric_df[xcol], numeric_df[ycol], alpha=.55, color="#34764d")
            ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f"{xcol} vs {ycol}")
            st.pyplot(fig)
            plt.close(fig)

with tabs[2]:
    st.subheader("Preprocessing inspection")
    target_col = st.selectbox("Target column", list(df.columns),
                              index=list(df.columns).index(target_guess), key="prep_target")
    feature_df = df.drop(columns=[target_col]).select_dtypes(include=np.number)
    st.write("Numeric feature columns available to the model:")
    st.write(", ".join(feature_df.columns) if len(feature_df.columns) else "No numeric columns found.")
    missing = df.isna().sum()
    st.dataframe(pd.DataFrame({
        "Missing values": missing,
        "Missing %": (missing / max(len(df), 1) * 100).round(2)
    }).sort_values("Missing %", ascending=False), use_container_width=True)
    st.markdown("""
    **Preprocessing applied during training**
    - Numeric columns are selected for the baseline model.
    - Missing numeric values are imputed using the median.
    - For common Pima dataset fields, medically implausible zeros are treated as missing.
    - Features are standardized inside a pipeline to prevent data leakage.
    """)

with tabs[3]:
    st.subheader("Train and evaluate model")
    target_col = st.selectbox("Choose prediction target", list(df.columns),
                              index=list(df.columns).index(target_guess), key="model_target")
    st.caption("For the classic Pima dataset, the target is usually `Outcome` (0 = negative, 1 = positive).")
    if st.button("🚀 Train model", type="primary"):
        try:
            result = fit_model(df, target_col, model_choice)
            model, feature_names, X_train, X_test, y_train, y_test, pred, metrics, encoder = result
            st.session_state.model_state = {
                "model": model, "features": feature_names, "target": target_col,
                "metrics": metrics, "X_test": X_test, "y_test": y_test, "pred": pred,
                "model_name": model_choice, "encoder": encoder,
            }
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Could not train model: {e}")
    state = st.session_state.model_state
    if state:
        cols = st.columns(len(state["metrics"]))
        for col, (name, value) in zip(cols, state["metrics"].items()):
            col.metric(name, f"{value:.1%}")
        cm = confusion_matrix(state["y_test"], state["pred"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("**Classification report**")
        report = classification_report(state["y_test"], state["pred"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
        if state["model_name"] == "Random Forest":
            clf = state["model"].named_steps["classifier"]
            importances = pd.Series(clf.feature_importances_, index=state["features"]).sort_values()
            fig, ax = plt.subplots(figsize=(7, 4))
            importances.plot(kind="barh", ax=ax, color="#34764d")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
            plt.close(fig)
        st.caption("Metrics depend on the uploaded dataset and the selected target. They are not guaranteed results.")

with tabs[4]:
    st.subheader("Patient risk prediction")
    state = st.session_state.model_state
    if not state:
        st.warning("Train a model in the Modeling tab before making predictions.")
    else:
        st.write("Enter the patient's values below.")
        feature_names = state["features"]
        numeric_df = df[feature_names]
        input_values = {}
        cols = st.columns(2)
        for i, feature in enumerate(feature_names):
            series = pd.to_numeric(numeric_df[feature], errors="coerce").dropna()
            default = float(series.median()) if not series.empty else 0.0
            minv = float(series.min()) if not series.empty else 0.0
            maxv = float(series.max()) if not series.empty else max(default + 1, 1.0)
            step = 0.1 if feature.lower() in ["bmi", "mass", "diabetespedigreefunction"] else 1.0
            with cols[i % 2]:
                input_values[feature] = st.number_input(
                    feature, min_value=minv, max_value=max(maxv, minv + step),
                    value=min(max(default, minv), max(maxv, minv + step)),
                    step=step, key=f"patient_{feature}"
                )
        if st.button("🔍 Predict diabetes", type="primary"):
            row = pd.DataFrame([input_values], columns=feature_names)
            prediction = int(state["model"].predict(row)[0])
            try:
                probability = float(state["model"].predict_proba(row)[0][1])
            except Exception:
                probability = None
            if prediction == 1:
                st.error("Model result: higher likelihood of diabetes")
            else:
                st.success("Model result: lower likelihood of diabetes")
            if probability is not None:
                st.progress(min(max(probability, 0.0), 1.0))
                st.metric("Estimated model probability for positive class", f"{probability:.1%}")
            st.warning("This is an educational machine-learning estimate, not a diagnosis. Consult a qualified healthcare professional for medical decisions.")

st.markdown("---")
st.caption("Diabetes Detection AI • Academic project • Streamlit interface")
