"""English Streamlit interface for three hospital-acquired infection models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Hospital-Acquired Infection Prediction Tool",
    page_icon="H",
    layout="wide",
)

DATA_ROOT = Path(r"D:\研究生学习\小论文写作\小论文数据\医院获得性")

MODEL_CONFIG = {
    "Model 1": {
        "title": "Model 1: Gram-Negative Infection Prediction",
        "outcome_label": "Predicted probability of Gram-negative infection",
        "directory": DATA_ROOT / "1" / "Final_Models",
        "prefix": "XGB",
        "fields": [
            ("WBC0", "White blood cell count on Day 0", "x10^9/L", 0.1),
            ("Neu0", "Neutrophil count on Day 0", "x10^9/L", 0.1),
            ("Potassium7", "Most recent potassium within the preceding 3 days", "mmol/L", 0.1),
            ("TP0", "Total protein on Day 0", "g/L", 0.1),
            ("PCT0", "Procalcitonin on Day 0", "ng/mL", 0.01),
            ("G-rate", "Department Gram-negative detection rate in the preceding year", "proportion", 0.001),
        ],
    },
    "Model 2": {
        "title": "Model 2: Drug-Resistant Organism Infection Prediction",
        "outcome_label": "Predicted probability of drug-resistant organism infection",
        "directory": DATA_ROOT / "2" / "Final_Models",
        "prefix": "RF",
        "fields": [
            ("Mechanical Ventilation", "Mechanical ventilation duration", "0 = none; 1 = 24-72 h; 2 = >72 h", 1, "ventilation"),
            ("Number of antibiotics", "Number of antibiotics", "count", 1),
            ("Hb DAY0", "Hemoglobin on Day 0", "g/L", 0.1),
            ("Hb prior", "Most recent hemoglobin within the preceding 3 days", "g/L", 0.1),
            ("WBC DAY0", "White blood cell count on Day 0", "x10^9/L", 0.1),
            ("MCV DAY0", "Mean corpuscular volume on Day 0", "fL", 0.1),
            ("Sodium prior", "Most recent sodium within the preceding 3 days", "mmol/L", 0.1),
            ("Cr DAY0", "Creatinine on Day 0", "umol/L", 0.1),
            ("Urea prior", "Most recent urea within the preceding 3 days", "mmol/L", 0.1),
            ("Urea DAY0", "Urea on Day 0", "mmol/L", 0.1),
            ("TP prior", "Most recent total protein within the preceding 3 days", "g/L", 0.1),
            ("TP DAY0", "Total protein on Day 0", "g/L", 0.1),
            ("PCT DAY0", "Procalcitonin on Day 0", "ng/mL", 0.01),
            ("MRDO rate", "Department multidrug-resistant organism detection rate in the preceding year", "proportion", 0.001),
        ],
    },
    "Model 3": {
        "title": "Model 3: CRO Infection Prediction",
        "outcome_label": "Predicted probability of CRO infection",
        "directory": DATA_ROOT / "3" / "Final_Models",
        "prefix": "RF",
        "fields": [
            ("Mechanical Ventilation", "Mechanical ventilation duration", "0 = none; 1 = 24-72 h; 2 = >72 h", 1, "ventilation"),
            ("Number of antibiotics", "Number of antibiotics", "count", 1),
            ("Number of Critical Values", "Number of critical laboratory values", "count", 1),
            ("Hb0", "Hemoglobin on Day 0", "g/L", 0.1),
            ("PLT0", "Platelet count on Day 0", "x10^9/L", 0.1),
            ("Urea7", "Most recent urea within the preceding 7 days", "mmol/L", 0.1),
            ("Urea0", "Urea on Day 0", "mmol/L", 0.1),
            ("ALB0", "Albumin on Day 0", "g/L", 0.1),
        ],
    },
}


@st.cache_resource(show_spinner="Loading model components...")
def load_components(directory: str, prefix: str):
    directory_path = Path(directory)
    with (directory_path / f"{prefix}_feature_names.json").open(encoding="utf-8") as file:
        feature_names = json.load(file)
    imputer = joblib.load(directory_path / f"{prefix}_MICE_imputer.joblib")
    scaler = joblib.load(directory_path / f"{prefix}_scaler.joblib")
    model = joblib.load(directory_path / f"{prefix}_model.joblib")
    return feature_names, imputer, scaler, model


def numeric_input(field: tuple):
    key, label, unit, step, *kind = field
    suffix = f" ({unit})" if unit else ""
    if kind and kind[0] == "ventilation":
        return st.selectbox(
            f"{label}{suffix}",
            options=[None, 0, 1, 2],
            format_func=lambda value: "Not available" if value is None else str(value),
            key=f"input_{key}",
        )
    return st.number_input(
        f"{label}{suffix}",
        value=None,
        step=float(step),
        placeholder="Leave blank if unavailable",
        key=f"input_{key}",
    )


def predict(values: dict, feature_names: list, imputer, scaler, model) -> float:
    frame = pd.DataFrame([{name: values.get(name, np.nan) for name in feature_names}])
    imputed = pd.DataFrame(imputer.transform(frame), columns=feature_names)
    scaled = scaler.transform(imputed)
    class_one_index = list(model.classes_).index(1)
    return float(model.predict_proba(scaled)[0, class_one_index])


st.markdown(
    """
    <style>
    .main-title { font-size: 36px; font-weight: 750; color: #163d5c; margin-bottom: 4px; }
    .subtitle { font-size: 17px; color: #52606d; margin-bottom: 24px; }
    .result { background: #edf7f1; border: 1px solid #b9dfc6; border-radius: 8px; padding: 22px; text-align: center; }
    .result-label { font-size: 17px; color: #285c3c; }
    .result-value { font-size: 42px; font-weight: 700; color: #176b3a; }
    </style>
    <div class="main-title">Hospital-Acquired Infection Prediction Tool</div>
    <div class="subtitle">Machine-learning based clinical risk estimation</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Prediction model")
selected_name = st.sidebar.radio("Choose a model", list(MODEL_CONFIG))
config = MODEL_CONFIG[selected_name]
st.sidebar.info(config["title"])

try:
    feature_names, imputer, scaler, model = load_components(str(config["directory"]), config["prefix"])
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

st.header(config["title"])
with st.expander("Data definitions and missing values", expanded=True):
    st.markdown(
        """
**Day 0** is the day on which the blood culture was collected. A **prior** measurement is the most recent test result obtained within the 3 days before Day 0.

All fields are optional. Missing values are processed using the model's trained MICE imputer before prediction.
"""
    )

st.caption("Enter available clinical and laboratory data. Leave a field blank when the value is unavailable.")
with st.form(f"{selected_name}_form"):
    left_column, right_column = st.columns(2)
    split_point = (len(config["fields"]) + 1) // 2
    values = {}
    with left_column:
        for field in config["fields"][:split_point]:
            values[field[0]] = numeric_input(field)
    with right_column:
        for field in config["fields"][split_point:]:
            values[field[0]] = numeric_input(field)
    submitted = st.form_submit_button("Calculate predicted probability", use_container_width=True)

if submitted:
    try:
        probability = predict(values, feature_names, imputer, scaler, model)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
    else:
        st.subheader("Prediction result")
        st.markdown(
            f'<div class="result"><div class="result-label">{config["outcome_label"]}</div>'
            f'<div class="result-value">{probability:.2%}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("This estimate is intended to support clinical research and does not replace clinical judgment.")
