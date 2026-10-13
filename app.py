import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide"
)

# =====================================================
# 2. PROJECT PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "ChargingRecords_cleaned.csv"
)

REGRESSION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "regression", "random_forest_regressor.pkl"
)

CLASSIFICATION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "classification", "random_forest_classifier.pkl"
)

FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

SHAP_IMPORTANCE_PATH = os.path.join(
    FIGURES_DIR, "shap_feature_importance.png"
)

SHAP_BEESWARM_PATH = os.path.join(
    FIGURES_DIR, "shap_beeswarm.png"
)

LIME_IMAGE_PATH = os.path.join(
    FIGURES_DIR, "lime_explanation.png"
)

LIME_HTML_PATH = os.path.join(
    FIGURES_DIR, "lime_explanation.html"
)

# =====================================================
# 3. CUSTOM STYLING
# =====================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 4. LOAD DATA AND MODELS
# =====================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    for column in ["StartDatetime", "EndDatetime"]:
        if column in data.columns:
            data[column] = pd.to_datetime(
                data[column], errors="coerce"
            )

    for column in [
        "Demand",
        "Duration",
        "UserID",
        "ChargerID",
        "ChargerCompany",
        "ChargerType"
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


@st.cache_resource
def load_regression_model():
    return joblib.load(REGRESSION_MODEL_PATH)


@st.cache_resource
def load_classification_model():
    return joblib.load(CLASSIFICATION_MODEL_PATH)


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# =====================================================
# 5. DASHBOARD HEADER
# =====================================================
st.title("⚡ EV Charging Demand Prediction")

st.markdown(
    "### Usage Pattern Analysis and Machine Learning Dashboard"
)

st.caption(
    "Analyze EV charging usage, predict demand, compare models, "
    "and explore model explanations."
)

st.divider()

# =====================================================
# 6. SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🔎 Dashboard Filters")

locations = sorted(df["Location"].dropna().unique().tolist())
charger_types = sorted(df["ChargerType"].dropna().unique().tolist())
companies = sorted(df["ChargerCompany"].dropna().unique().tolist())

selected_locations = st.sidebar.multiselect(
    "Select Location",
    locations,
    default=locations
)

selected_types = st.sidebar.multiselect(
    "Select Charger Type",
    charger_types,
    default=charger_types
)

selected_companies = st.sidebar.multiselect(
    "Select Charger Company",
    companies,
    default=companies
)

valid_dates = df["StartDatetime"].dropna()

if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_dates = ()

filtered_df = df[
    df["Location"].isin(selected_locations)
    & df["ChargerType"].isin(selected_types)
    & df["ChargerCompany"].isin(selected_companies)
].copy()

if (
    isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
):
    start_date, end_date = selected_dates

    filtered_df = filtered_df[
        filtered_df["StartDatetime"].dt.date.between(
            start_date, end_date
        )
    ]

# =====================================================
# 7. DATASET OVERVIEW
# =====================================================
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Charging Records", f"{len(filtered_df):,}")

with col2:
    st.metric("Unique Users", f"{filtered_df['UserID'].nunique():,}")

with col3:
    st.metric(
        "Charging Locations",
        f"{filtered_df['Location'].nunique():,}"
    )

with col4:
    avg_demand = filtered_df["Demand"].mean()

    st.metric(
        "Average Demand",
        f"{avg_demand:.2f}" if pd.notna(avg_demand) else "N/A"
    )

if filtered_df.empty:
    st.warning("No records match your selected filters.")
    st.stop()

st.divider()

# =====================================================
# 8. CHARGING DEMAND ANALYSIS
# =====================================================
st.subheader("📈 Charging Demand Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Average Demand by Location")

    location_demand = (
        filtered_df.groupby("Location")["Demand"]
        .mean()
        .dropna()
        .sort_values()
    )

    st.bar_chart(location_demand)

with col2:
    st.markdown("#### Average Demand by Charger Type")

    charger_demand = (
        filtered_df.groupby("ChargerType")["Demand"]
        .mean()
        .dropna()
        .sort_index()
    )

    st.bar_chart(charger_demand)

st.markdown("#### Demand Distribution")

demand_values = filtered_df["Demand"].dropna()

if not demand_values.empty:
    counts, edges = np.histogram(demand_values, bins=20)

    labels = [
        f"{edges[i]:.1f}–{edges[i + 1]:.1f}"
        for i in range(len(edges) - 1)
    ]

    histogram_df = pd.DataFrame({
        "Demand Range": labels,
        "Number of Records": counts
    }).set_index("Demand Range")

    st.bar_chart(histogram_df)

st.divider()

# =====================================================
# 9. CHARGING USAGE PATTERNS
# =====================================================
st.subheader("🕒 Charging Usage Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Monthly Average Demand")

    monthly_data = filtered_df.dropna(
        subset=["StartDatetime", "Demand"]
    ).copy()

    if not monthly_data.empty:
        monthly_data["Month"] = (
            monthly_data["StartDatetime"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_demand = (
            monthly_data.groupby("Month")["Demand"]
            .mean()
            .sort_index()
        )

        st.line_chart(monthly_demand)
    else:
        st.info("No valid date records available.")

with col2:
    st.markdown("#### Demand by Charging Duration")

    duration_data = filtered_df.dropna(
        subset=["Duration", "Demand"]
    ).copy()

    duration_data = duration_data[
        duration_data["Duration"] >= 0
    ]

    if not duration_data.empty:
        duration_limit = duration_data["Duration"].quantile(0.95)

        duration_data = duration_data[
            duration_data["Duration"] <= duration_limit
        ].copy()

        if not duration_data.empty:
            duration_data["Duration Range"] = pd.cut(
                duration_data["Duration"],
                bins=8,
                duplicates="drop"
            )

            duration_demand = (
                duration_data.groupby(
                    "Duration Range", observed=True
                )["Demand"]
                .mean()
                .dropna()
            )

            duration_demand.index = [
                f"{interval.left:.0f}–{interval.right:.0f} min"
                for interval in duration_demand.index
            ]

            st.bar_chart(duration_demand)
        else:
            st.info("No valid duration values available.")
    else:
        st.info("No valid duration data available.")

st.divider()

# =====================================================
# 10. PREDICTION HELPERS
# =====================================================
def numeric_options(series):
    return sorted(
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )


def prepare_prediction_features(
    user_id,
    charger_id,
    company,
    location,
    charger_type,
    start_day,
    duration,
    model
):
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The saved model does not expose feature_names_in_. "
            "Check the original training pipeline."
        )

    input_df = pd.DataFrame([{
        "UserID": int(user_id),
        "ChargerID": int(charger_id),
        "ChargerCompany": company,
        "Location": str(location),
        "ChargerType": charger_type,
        "StartDay": str(start_day),
        "Duration": float(duration)
    }])

    encoded_df = pd.get_dummies(
        input_df,
        columns=["Location", "StartDay"],
        dtype=int
    )

    expected_features = list(model.feature_names_in_)

    missing_features = [
        feature for feature in expected_features
        if feature not in encoded_df.columns
        and not feature.startswith("Location_")
        and not feature.startswith("StartDay_")
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
            + ". Update the form to match the original training pipeline."
        )

    return encoded_df.reindex(
        columns=expected_features,
        fill_value=0
    )


company_values = numeric_options(df["ChargerCompany"])
type_values = numeric_options(df["ChargerType"])
location_options = sorted(df["Location"].dropna().unique().tolist())
day_options = sorted(df["StartDay"].dropna().astype(str).unique().tolist())

if not company_values or not type_values or not location_options:
    st.error("Could not determine prediction options from the dataset.")
    st.stop()

if not day_options:
    st.error("StartDay values are missing from the dataset.")
    st.stop()

default_user_id = max(0, int(df["UserID"].median()))
default_charger_id = max(0, int(df["ChargerID"].median()))
default_duration = max(0.0, float(df["Duration"].median()))

# =====================================================
# 11. MACHINE LEARNING PREDICTIONS
# =====================================================
st.header("🤖 Machine Learning Predictions")

tab_regression, tab_classification = st.tabs([
    "📉 Regression Prediction",
    "🚦 Demand Classification"
])

# -----------------------------------------------------
# 11A. REGRESSION PREDICTION
# -----------------------------------------------------
with tab_regression:
    st.subheader("Predict Numerical Charging Demand")

    with st.form("regression_form"):
        col1, col2 = st.columns(2)

        with col1:
            reg_user_id = st.number_input(
                "User ID", min_value=0,
                value=default_user_id, step=1
            )

            reg_charger_id = st.number_input(
                "Charger ID", min_value=0,
                value=default_charger_id, step=1
            )

            reg_location = st.selectbox(
                "Location", location_options
            )

            reg_day = st.selectbox(
                "Start Day", day_options
            )

        with col2:
            reg_company = st.selectbox(
                "Charger Company", company_values
            )

            reg_type = st.selectbox(
                "Charger Type", type_values
            )

            reg_duration = st.number_input(
                "Charging Duration (minutes)",
                min_value=0.0,
                value=default_duration,
                step=1.0
            )

        regression_submit = st.form_submit_button(
            "Predict Charging Demand",
            type="primary"
        )

    if regression_submit:
        try:
            model = load_regression_model()

            features = prepare_prediction_features(
                reg_user_id,
                reg_charger_id,
                reg_company,
                reg_location,
                reg_type,
                reg_day,
                reg_duration,
                model
            )

            prediction = float(model.predict(features)[0])

            st.success("Regression prediction completed.")
            st.metric("Predicted Charging Demand", f"{prediction:.2f}")

        except Exception as error:
            st.error(f"Regression prediction failed: {error}")

# -----------------------------------------------------
# 11B. CLASSIFICATION PREDICTION
# -----------------------------------------------------
with tab_classification:
    st.subheader("Classify Charging Demand")

    st.caption(
        "Class 0 = Normal Demand; Class 1 = High Demand. "
        "The classification threshold used in the project is 14.1."
    )

    with st.form("classification_form"):
        col1, col2 = st.columns(2)

        with col1:
            cls_user_id = st.number_input(
                "User ID", min_value=0,
                value=default_user_id, step=1
            )

            cls_charger_id = st.number_input(
                "Charger ID", min_value=0,
                value=default_charger_id, step=1
            )

            cls_location = st.selectbox(
                "Location", location_options
            )

            cls_day = st.selectbox(
                "Start Day", day_options
            )

        with col2:
            cls_company = st.selectbox(
                "Charger Company", company_values
            )

            cls_type = st.selectbox(
                "Charger Type", type_values
            )

            cls_duration = st.number_input(
                "Charging Duration (minutes)",
                min_value=0.0,
                value=default_duration,
                step=1.0
            )

        classification_submit = st.form_submit_button(
            "Predict Demand Level",
            type="primary"
        )

    if classification_submit:
        try:
            model = load_classification_model()

            features = prepare_prediction_features(
                cls_user_id,
                cls_charger_id,
                cls_company,
                cls_location,
                cls_type,
                cls_day,
                cls_duration,
                model
            )

            predicted_class = int(model.predict(features)[0])

            probabilities = None
            model_classes = list(model.classes_)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]

            st.session_state["latest_classification"] = {
                "predicted_class": predicted_class,
                "probabilities": (
                    probabilities.tolist()
                    if probabilities is not None else None
                ),
                "classes": model_classes,
                "features": features.iloc[0].to_dict()
            }

            if predicted_class == 1:
                st.success("Predicted Demand Level: HIGH DEMAND")
            else:
                st.info("Predicted Demand Level: NORMAL DEMAND")

            if probabilities is not None:
                normal_probability = (
                    float(probabilities[model_classes.index(0)])
                    if 0 in model_classes else None
                )

                high_probability = (
                    float(probabilities[model_classes.index(1)])
                    if 1 in model_classes else None
                )

                col1, col2 = st.columns(2)

                with col1:
                    if normal_probability is not None:
                        st.metric(
                            "Normal Demand",
                            f"{normal_probability:.1%}"
                        )

                with col2:
                    if high_probability is not None:
                        st.metric(
                            "High Demand",
                            f"{high_probability:.1%}"
                        )

        except Exception as error:
            st.error(f"Classification prediction failed: {error}")

st.divider()

# =====================================================
# 12. REGRESSION MODEL PERFORMANCE
# =====================================================
st.header("📉 Regression Model Performance")

regression_metrics = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "MAE": 6.2987,
        "MSE": 102.5193,
        "RMSE": 10.1252,
        "R²": 0.4311
    },
    {
        "Model": "Decision Tree",
        "MAE": 2.9266,
        "MSE": 43.0029,
        "RMSE": 6.5577,
        "R²": 0.7614
    },
    {
        "Model": "Random Forest",
        "MAE": 2.4807,
        "MSE": 25.3661,
        "RMSE": 5.0365,
        "R²": 0.8592
    },
    {
        "Model": "Gradient Boosting",
        "MAE": 3.6254,
        "MSE": 39.8639,
        "RMSE": 6.3138,
        "R²": 0.7788
    },
    {
        "Model": "XGBoost",
        "MAE": 2.9484,
        "MSE": 29.2645,
        "RMSE": 5.4097,
        "R²": 0.8376
    }
])

st.dataframe(
    regression_metrics.style.format({
        "MAE": "{:.4f}",
        "MSE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

reg_metric_choice = st.selectbox(
    "Compare regression models using",
    ["RMSE", "MAE", "MSE", "R²"]
)

reg_chart = regression_metrics.set_index("Model")[
    reg_metric_choice
].sort_values(ascending=(reg_metric_choice != "R²"))

st.bar_chart(reg_chart)

best_regression = regression_metrics.sort_values(
    reg_metric_choice,
    ascending=(reg_metric_choice != "R²")
).iloc[0]

st.success(
    f"Best model by {reg_metric_choice}: "
    f"{best_regression['Model']} "
    f"({best_regression[reg_metric_choice]:.4f})"
)

st.divider()

# =====================================================
# 13. CLASSIFICATION MODEL PERFORMANCE
# =====================================================
st.header("🚦 Classification Model Performance")

classification_metrics = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.8320,
        "ROC-AUC": 0.9028
    },
    {
        "Model": "K-Nearest Neighbors",
        "Accuracy": 0.7701,
        "ROC-AUC": 0.8434
    },
    {
        "Model": "Linear SVC",
        "Accuracy": 0.7971,
        "ROC-AUC": 0.8776
    },
    {
        "Model": "Decision Tree",
        "Accuracy": 0.8829,
        "ROC-AUC": 0.9293
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.8666,
        "ROC-AUC": 0.9363
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": 0.8701,
        "ROC-AUC": 0.9262
    }
])

st.dataframe(
    classification_metrics.style.format({
        "Accuracy": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

classification_metric_choice = st.selectbox(
    "Compare classification models using",
    ["Accuracy", "ROC-AUC"]
)

cls_chart = classification_metrics.set_index("Model")[
    classification_metric_choice
].sort_values()

st.bar_chart(cls_chart)

best_classifier = classification_metrics.loc[
    classification_metrics[classification_metric_choice].idxmax()
]

st.success(
    f"Best recorded model by {classification_metric_choice}: "
    f"{best_classifier['Model']} "
    f"({best_classifier[classification_metric_choice]:.4f})"
)

st.warning(
    "These are recorded experiment results. Confirm that the saved "
    "classifier matches the experiment before reporting these metrics."
)

st.divider()

# =====================================================
# 14. SHAP EXPLANATIONS
# =====================================================
st.header("🔍 Model Explainability")

st.subheader("SHAP Feature Importance")

if os.path.isfile(SHAP_IMPORTANCE_PATH):
    st.image(
        SHAP_IMPORTANCE_PATH,
        caption="SHAP Feature Importance",
        use_container_width=True
    )
else:
    st.warning(
        f"SHAP image not found: {SHAP_IMPORTANCE_PATH}"
    )

st.subheader("SHAP Beeswarm Plot")

if os.path.isfile(SHAP_BEESWARM_PATH):
    st.image(
        SHAP_BEESWARM_PATH,
        caption="SHAP Beeswarm Plot",
        use_container_width=True
    )
else:
    st.warning(
        f"SHAP image not found: {SHAP_BEESWARM_PATH}"
    )

st.divider()

# =====================================================
# 15. LIME: EMBED INTERACTIVE CHART ON THE SAME PAGE
# =====================================================
st.subheader("🧠 LIME Local Explanation")

if os.path.isfile(LIME_HTML_PATH):
    try:
        with open(LIME_HTML_PATH, "r", encoding="utf-8") as file:
            lime_html = file.read()

        # Inject compact, light styling into the saved HTML.
        light_css = """
        <style>
        html, body {
            background: #ffffff !important;
            color: #222222 !important;
            margin: 0 !important;
            padding: 8px !important;
            font-family: Arial, sans-serif !important;
            overflow-x: hidden !important;
        }

        body {
            color-scheme: light !important;
        }

        svg, canvas {
            max-width: 100% !important;
        }

        </style>
        """

        head_position = lime_html.lower().find("</head>")

        if head_position != -1:
            lime_html = (
                lime_html[:head_position]
                + light_css
                + lime_html[head_position:]
            )
        else:
            lime_html = light_css + lime_html

        # Display the actual interactive LIME HTML inside Streamlit.
        components.html(
            lime_html,
            height=420,
            scrolling=True
        )

    except Exception as error:
        st.error(
            f"Unable to display the interactive LIME chart: {error}"
        )

else:
    st.warning(
        f"LIME HTML file not found: {LIME_HTML_PATH}"
    )

# Compact saved LIME image below the interactive chart.
if os.path.isfile(LIME_IMAGE_PATH):
    with st.expander("View static LIME explanation"):
        st.image(
            LIME_IMAGE_PATH,
            caption="Saved LIME Explanation",
            use_container_width=True
        )

st.caption(
    "The saved LIME explanation is a previously generated explanation. "
    "It does not automatically recalculate for every new prediction."
)

st.divider()

# =====================================================
# 16. DATA PREVIEW AND DOWNLOAD
# =====================================================
st.subheader("📄 Filtered Dataset Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv_data,
    file_name="ev_charging_filtered_data.csv",
    mime="text/csv"
)

st.divider()

# =====================================================
# 17. FOOTER
# =====================================================
st.caption(
    "EV Charging Demand Prediction and Usage Pattern Analysis "
    "| Machine Learning Project"
)