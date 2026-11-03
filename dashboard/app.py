
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "ChargingRecords_cleaned.csv"

REGRESSION_MODEL_PATH = (
    BASE_DIR / "models" / "regression" / "random_forest_regressor.pkl"
)

CLASSIFICATION_MODEL_PATH = (
    BASE_DIR / "models" / "classification" / "random_forest_classifier.pkl"
)

FIGURES_DIR = BASE_DIR / "reports" / "figures"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ===== MAIN PAGE ===== */

.stApp {
    background-color: #F4F7FC;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F7FC;
}

[data-testid="stHeader"] {
    background-color: transparent;
}

/* Main page text */
[data-testid="stAppViewContainer"] .stMarkdown,
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stCaption {
    color: #263449;
}

/* Headings */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5 {
    color: #14213D !important;
    font-weight: 700;
}

/* ===== SIDEBAR ===== */

[data-testid="stSidebar"] {
    background-color: #101B35;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] input {
    color: #14213D !important;
    background-color: #FFFFFF !important;
}

/* ===== HEADER ===== */

.main-header {
    background: linear-gradient(115deg, #14213D, #183D63);
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 8px;
}

.main-header p {
    color: #E0EAF7 !important;
    font-size: 14px;
    margin-bottom: 0;
}

/* ===== SECTION HEADINGS ===== */

.section-heading {
    color: #14213D !important;
    font-size: 22px;
    font-weight: 750;
    margin-top: 14px;
    margin-bottom: 15px;
}

.section-description {
    color: #64748B !important;
    font-size: 13px;
    margin-top: -10px;
    margin-bottom: 18px;
}

/* ===== KPI CARDS ===== */

.kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EF;
    border-radius: 15px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(20, 33, 61, 0.05);
}

.kpi-label {
    color: #64748B !important;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
}

.kpi-value {
    color: #14213D !important;
    font-size: 26px;
    font-weight: 800;
}

.kpi-note {
    color: #087E75 !important;
    font-size: 11px;
    margin-top: 5px;
}

/* ===== INFO CARDS ===== */

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EF;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 12px;
}

.info-card h4 {
    color: #475569 !important;
}

/* ===== PREDICTION RESULT ===== */

.prediction-result {
    background-color: #E5F6F3;
    border: 1px solid #A9DED5;
    border-radius: 14px;
    padding: 20px;
    margin-top: 15px;
}

.prediction-result h3 {
    color: #087E75 !important;
    margin-bottom: 5px;
}

.prediction-result p {
    color: #315D59 !important;
}

/* ===== STREAMLIT METRICS ===== */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE4EF;
    border-radius: 13px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #14213D !important;
}

[data-testid="stMetricDelta"] {
    color: #087E75 !important;
}

/* ===== INPUT FIELDS ===== */

.stTextInput input,
.stNumberInput input,
.stDateInput input {
    color: #14213D !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border-color: #CBD5E1 !important;
}

[data-baseweb="select"] * {
    color: #14213D !important;
}

/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] * {
    color: #14213D !important;
}

/* ===== TABS ===== */

[data-testid="stTabs"] button {
    color: #475569 !important;
    font-weight: 600;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #087E75 !important;
    border-bottom-color: #0F9D92 !important;
}

/* ===== DATA TABLES ===== */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
}

[data-testid="stTable"] {
    background-color: #FFFFFF;
}

/* ===== BUTTONS ===== */

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background-color: #0F9D92 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px;
    font-weight: 650;
    padding: 10px 18px;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background-color: #087E75 !important;
    color: #FFFFFF !important;
}

/* ===== ALERTS ===== */

[data-testid="stAlert"] {
    border-radius: 10px;
}

[data-testid="stAlert"] p {
    color: #263449 !important;
}

/* ===== FOOTER ===== */

.footer {
    text-align: center;
    color: #64748B !important;
    font-size: 12px;
    padding: 25px 0 10px;
}

.footer b {
    color: #14213D !important;
}
</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    for col in ["Demand", "Duration", "UserID", "ChargerID",
                "ChargerCompany", "ChargerType"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_candidates = [
        "StartDate", "Start_Date", "Date", "StartTime", "Start_Time"
    ]

    for col in date_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            break

    return df


@st.cache_resource
def load_models():
    regression_model = None
    classification_model = None

    if REGRESSION_MODEL_PATH.exists():
        regression_model = joblib.load(REGRESSION_MODEL_PATH)

    if CLASSIFICATION_MODEL_PATH.exists():
        classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)

    return regression_model, classification_model


try:
    df = load_data()
    regression_model, classification_model = load_models()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_number(value):
    return f"{value:,.0f}"


def get_unique_options(column):
    if column not in df.columns:
        return []

    return sorted(df[column].dropna().unique().tolist())


def get_date_column(data):
    candidates = [
        "StartDate", "Start_Date", "Date", "StartTime", "Start_Time"
    ]

    for col in candidates:
        if col in data.columns and pd.api.types.is_datetime64_any_dtype(data[col]):
            return col

    return None


def get_model_features(model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


def prepare_prediction_features(model, values):
    """
    Prepare input features in the exact order expected by the saved model.

    Supports:
    - Numeric columns
    - One-hot encoded Location columns
    - One-hot encoded StartDay columns
    """

    feature_names = get_model_features(model)

    if not feature_names:
        raise ValueError(
            "The saved model does not expose feature_names_in_. "
            "The original training feature order must be supplied."
        )

    row = {feature: 0 for feature in feature_names}

    # Assign ordinary numeric features when present in the model.
    for key, value in values.items():
        if key in row:
            row[key] = value

    # Match one-hot encoded Location columns.
    location_value = str(values.get("Location", ""))

    location_candidates = [
        f"Location_{location_value}",
        f"Location_{values.get('Location', '')}"
    ]

    for feature in location_candidates:
        if feature in row:
            row[feature] = 1

    # Match one-hot encoded StartDay columns.
    day_value = str(values.get("StartDay", ""))

    day_candidates = [
        f"StartDay_{day_value}",
        f"StartDay_{values.get('StartDay', '')}"
    ]

    for feature in day_candidates:
        if feature in row:
            row[feature] = 1

    return pd.DataFrame([row], columns=feature_names)


def render_kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, description=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True
        )


def get_day_options():
    if "StartDay" in df.columns:
        return get_unique_options("StartDay")

    return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 10px 0 22px 0;">
            <h2 style="color:white; margin-bottom:2px;">⚡ EV Analytics</h2>
            <p style="color:#AEBBD2; font-size:12px;">
                Charging Demand Prediction
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Dashboard Filters")

    filtered_df = df.copy()

    for column in ["Location", "ChargerType", "ChargerCompany"]:

        if column in filtered_df.columns:

            options = get_unique_options(column)

            selected = st.multiselect(
                column,
                options=options,
                default=options,
                key=f"filter_{column}"
            )

            if selected:
                filtered_df = filtered_df[
                    filtered_df[column].isin(selected)
                ]
            else:
                filtered_df = filtered_df.iloc[0:0]

    date_column = get_date_column(filtered_df)

    if date_column and not filtered_df.empty:

        valid_dates = filtered_df[date_column].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:

                start_date, end_date = date_range

                filtered_df = filtered_df[
                    (filtered_df[date_column].dt.date >= start_date)
                    & (filtered_df[date_column].dt.date <= end_date)
                ]

    st.markdown("---")

    st.caption("EV Charging Demand Prediction")
    st.caption("Machine Learning Analytics Dashboard")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>EV Charging Demand Dashboard</h1>
        <p>
            Explore charging patterns, predict demand, compare machine
            learning models, and understand model decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# OVERVIEW KPI SECTION
# ============================================================

section_title(
    "Project Overview",
    "A summary of the charging records based on your selected filters."
)

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

total_records = len(filtered_df)

unique_users = (
    filtered_df["UserID"].nunique()
    if "UserID" in filtered_df.columns else 0
)

unique_locations = (
    filtered_df["Location"].nunique()
    if "Location" in filtered_df.columns else 0
)

average_demand = (
    filtered_df["Demand"].mean()
    if "Demand" in filtered_df.columns else 0
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi(
        "TOTAL RECORDS",
        format_number(total_records),
        "Filtered charging records"
    )

with col2:
    render_kpi(
        "UNIQUE USERS",
        format_number(unique_users),
        "Customers in selected data"
    )

with col3:
    render_kpi(
        "LOCATIONS",
        format_number(unique_locations),
        "Charging locations"
    )

with col4:
    render_kpi(
        "AVERAGE DEMAND",
        f"{average_demand:.2f}" if pd.notna(average_demand) else "N/A",
        "Average recorded demand"
    )


# ============================================================
# DASHBOARD NAVIGATION
# ============================================================

st.markdown("---")

overview_tab, prediction_tab, performance_tab, explain_tab, data_tab = st.tabs(
    [
        "📊 Analytics",
        "🔮 Predictions",
        "📈 Model Performance",
        "🧠 Explainability",
        "📁 Dataset"
    ]
)


# ============================================================
# TAB 1: ANALYTICS
# ============================================================

with overview_tab:

    section_title(
        "Charging Demand Analytics",
        "Understand how charging demand changes across locations, "
        "charger types, and time."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown("#### Average Demand by Location")

        if {"Location", "Demand"}.issubset(filtered_df.columns):

            location_demand = (
                filtered_df.groupby("Location")["Demand"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(location_demand)

        else:
            st.info("Location or Demand column is unavailable.")

    with chart_col2:

        st.markdown("#### Average Demand by Charger Type")

        if {"ChargerType", "Demand"}.issubset(filtered_df.columns):

            type_demand = (
                filtered_df.groupby("ChargerType")["Demand"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(type_demand)

        else:
            st.info("ChargerType or Demand column is unavailable.")

    st.markdown("---")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:

        st.markdown("#### Demand Distribution")

        if "Demand" in filtered_df.columns:

            demand_data = filtered_df["Demand"].dropna()

            if not demand_data.empty:
                histogram = pd.cut(
                    demand_data,
                    bins=20
                ).value_counts().sort_index()

                histogram.index = histogram.index.astype(str)

                st.bar_chart(histogram)

        else:
            st.info("Demand column is unavailable.")

    with chart_col4:

        st.markdown("#### Monthly Average Demand")

        if date_column and "Demand" in filtered_df.columns:

            monthly_data = filtered_df.dropna(
                subset=[date_column, "Demand"]
            ).copy()

            if not monthly_data.empty:

                monthly_data["Month"] = (
                    monthly_data[date_column].dt.to_period("M").astype(str)
                )

                monthly_demand = (
                    monthly_data.groupby("Month")["Demand"]
                    .mean()
                    .sort_index()
                )

                st.line_chart(monthly_demand)

            else:
                st.info("No valid date records are available.")

        else:
            st.info("Date or Demand column is unavailable.")

    st.markdown("---")

    section_title("Charging Duration Analysis")

    if {"Duration", "Demand"}.issubset(filtered_df.columns):

        duration_data = filtered_df[
            ["Duration", "Demand"]
        ].dropna()

        if not duration_data.empty:

            duration_data = duration_data.copy()

            duration_data["Duration Group"] = pd.cut(
                duration_data["Duration"],
                bins=10
            )

            duration_demand = (
                duration_data.groupby(
                    "Duration Group",
                    observed=False
                )["Demand"].mean()
            )

            duration_demand.index = duration_demand.index.astype(str)

            st.bar_chart(duration_demand)

        else:
            st.info("No valid duration records are available.")

    else:
        st.info("Duration or Demand column is unavailable.")


# ============================================================
# TAB 2: PREDICTIONS
# ============================================================

with prediction_tab:

    section_title(
        "Charging Demand Prediction",
        "Enter charging details to estimate demand and classify the "
        "expected demand level."
    )

    if regression_model is None or classification_model is None:

        st.error(
            "One or both trained models are missing. "
            "Check your models folder."
        )

    else:

        locations = get_unique_options("Location")
        charger_types = get_unique_options("ChargerType")
        charger_companies = get_unique_options("ChargerCompany")
        day_options = get_day_options()

        if not locations or not charger_types or not charger_companies:

            st.warning(
                "Required input columns or values are missing from the dataset."
            )

        else:

            st.markdown("#### Charging Information")

            with st.form("prediction_form"):

                input_col1, input_col2 = st.columns(2)

                with input_col1:

                    user_id = st.number_input(
                        "User ID",
                        min_value=0,
                        value=1001,
                        step=1
                    )

                    charger_id = st.number_input(
                        "Charger ID",
                        min_value=0,
                        value=101,
                        step=1
                    )

                    location = st.selectbox(
                        "Location",
                        locations
                    )

                    charger_type = st.selectbox(
                        "Charger Type",
                        charger_types
                    )

                with input_col2:

                    charger_company = st.selectbox(
                        "Charger Company",
                        charger_companies
                    )

                    duration = st.number_input(
                        "Charging Duration",
                        min_value=0.0,
                        value=30.0,
                        step=1.0
                    )

                    if day_options:

                        start_day = st.selectbox(
                            "Start Day",
                            day_options
                        )

                    else:

                        start_day = st.selectbox(
                            "Start Day",
                            [
                                "Monday", "Tuesday", "Wednesday",
                                "Thursday", "Friday", "Saturday", "Sunday"
                            ]
                        )

                predict_button = st.form_submit_button(
                    "Predict Charging Demand",
                    use_container_width=True
                )

            if predict_button:

                input_values = {
                    "UserID": user_id,
                    "ChargerID": charger_id,
                    "ChargerCompany": charger_company,
                    "Location": location,
                    "ChargerType": charger_type,
                    "StartDay": start_day,
                    "Duration": duration
                }

                try:

                    regression_input = prepare_prediction_features(
                        regression_model,
                        input_values
                    )

                    classification_input = prepare_prediction_features(
                        classification_model,
                        input_values
                    )

                    regression_prediction = regression_model.predict(
                        regression_input
                    )[0]

                    classification_prediction = classification_model.predict(
                        classification_input
                    )[0]

                    st.markdown("---")

                    section_title("Prediction Results")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.markdown(
                            """
                            <div class="info-card">
                                <h4 style="color:#64748B;">
                                    Regression Prediction
                                </h4>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.metric(
                            "Predicted Demand",
                            f"{regression_prediction:.2f}"
                        )

                    with result_col2:

                        class_label = (
                            "HIGH DEMAND"
                            if int(classification_prediction) == 1
                            else "NORMAL DEMAND"
                        )

                        st.markdown(
                            """
                            <div class="info-card">
                                <h4 style="color:#64748B;">
                                    Classification Prediction
                                </h4>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"""
                            <div class="prediction-result">
                                <h3>{class_label}</h3>
                                <p>Predicted demand category</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if hasattr(classification_model, "predict_proba"):

                        probabilities = (
                            classification_model.predict_proba(
                                classification_input
                            )[0]
                        )

                        classes = list(classification_model.classes_)

                        probability_data = pd.DataFrame({
                            "Demand Category": [
                                "Normal Demand", "High Demand"
                            ],
                            "Probability": [
                                probabilities[classes.index(0)],
                                probabilities[classes.index(1)]
                            ]
                        })

                        st.markdown("#### Classification Probability")

                        probability_data = probability_data.set_index(
                            "Demand Category"
                        )

                        st.bar_chart(probability_data)

                    st.caption(
                        "The regression model estimates a numerical demand "
                        "value. The classification model independently "
                        "predicts the demand category."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

                    st.info(
                        "Check that the prediction inputs and one-hot encoded "
                        "features match the features used when training "
                        "the saved models."
                    )


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with performance_tab:

    section_title(
        "Machine Learning Model Evaluation",
        "Compare the models evaluated during project development."
    )

    st.markdown("#### Regression Model Comparison")

    regression_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "Gradient Boosting",
            "Decision Tree",
            "Linear Regression"
        ],
        "MAE": [
            2.4807,
            2.9484,
            3.6254,
            2.9266,
            6.2987
        ],
        "MSE": [
            25.3661,
            29.2645,
            39.8639,
            43.0029,
            102.5193
        ],
        "RMSE": [
            5.0365,
            5.4097,
            6.3138,
            6.5577,
            10.1252
        ],
        "R² Score": [
            0.8592,
            0.8376,
            0.7788,
            0.7614,
            0.4311
        ]
    })

    st.dataframe(
        regression_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("##### Regression R² Comparison")

    st.bar_chart(
        regression_results.set_index("Model")[["R² Score"]]
    )

    st.markdown("---")

    st.markdown("#### Classification Model Comparison")

    classification_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Tuned Random Forest",
            "Decision Tree",
            "Logistic Regression",
            "KNN",
            "Linear SVC"
        ],
        "Accuracy": [
            0.8666,
            0.8701,
            0.8829,
            0.8320,
            0.7701,
            0.7971
        ],
        "ROC-AUC": [
            0.9363,
            0.9262,
            0.9293,
            0.9028,
            0.8434,
            0.8776
        ]
    })

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("##### Classification Accuracy Comparison")

    st.bar_chart(
        classification_results.set_index("Model")[["Accuracy"]]
    )

    st.info(
        "The saved classification artifact is the tuned Random Forest model. "
        "The Decision Tree achieved the highest recorded accuracy in the "
        "model comparison. Results shown here are the previously recorded "
        "evaluation results."
    )

    st.markdown("---")

    st.markdown("#### Time-Series Forecasting")

    time_series_path = BASE_DIR / "reports" / "time_series_results.csv"

    if time_series_path.exists():

        try:

            time_series_results = pd.read_csv(time_series_path)

            st.dataframe(
                time_series_results,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.warning(f"Unable to display time-series results: {e}")

    else:

        st.info("Time-series results file was not found.")


# ============================================================
# TAB 4: EXPLAINABILITY
# ============================================================

with explain_tab:

    section_title(
        "Model Explainability",
        "Understand which features influence the model's predictions "
        "using SHAP and LIME."
    )

    shap_importance_path = (
        FIGURES_DIR / "shap_feature_importance.png"
    )

    shap_beeswarm_path = (
        FIGURES_DIR / "shap_beeswarm.png"
    )

    lime_image_path = (
        FIGURES_DIR / "lime_explanation.png"
    )

    lime_html_path = (
        FIGURES_DIR / "lime_explanation.html"
    )

    st.markdown("#### SHAP Analysis")

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:

        st.markdown("##### Feature Importance")

        if shap_importance_path.exists():

            st.image(
                str(shap_importance_path),
                use_container_width=True
            )

        else:

            st.info("SHAP feature importance image not found.")

    with shap_col2:

        st.markdown("##### SHAP Summary Plot")

        if shap_beeswarm_path.exists():

            st.image(
                str(shap_beeswarm_path),
                use_container_width=True
            )

        else:

            st.info("SHAP beeswarm image not found.")

    st.markdown("---")

    st.markdown("#### LIME Explanation")

    if lime_html_path.exists():

        try:

            html_content = lime_html_path.read_text(
                encoding="utf-8"
            )

            components.html(
                html_content,
                height=650,
                scrolling=True
            )

        except Exception as e:

            st.warning(f"Unable to load LIME HTML: {e}")

    elif lime_image_path.exists():

        st.image(
            str(lime_image_path),
            use_container_width=True
        )

    else:

        st.info("LIME explanation files were not found.")

    st.caption(
        "SHAP explains feature contributions across predictions. "
        "LIME provides a local explanation for an individual prediction."
    )


# ============================================================
# TAB 5: DATASET
# ============================================================

with data_tab:

    section_title(
        "Charging Dataset",
        "Explore the filtered dataset and download the selected records."
    )

    data_col1, data_col2, data_col3 = st.columns(3)

    with data_col1:
        st.metric("Rows", format_number(len(filtered_df)))

    with data_col2:
        st.metric("Columns", format_number(len(filtered_df.columns)))

    with data_col3:
        st.metric(
            "Missing Values",
            format_number(int(filtered_df.isnull().sum().sum()))
        )

    st.markdown("#### Dataset Preview")

    st.dataframe(
        filtered_df.head(500),
        use_container_width=True,
        height=420
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv_data,
        file_name="EV_Charging_Filtered_Data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")

    st.markdown("#### Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.markdown("**Column Names**")

        for column in filtered_df.columns:
            st.write(f"• {column}")

    with info_col2:

        st.markdown("**Data Types**")

        st.dataframe(
            pd.DataFrame({
                "Column": filtered_df.dtypes.index,
                "Data Type": filtered_df.dtypes.astype(str).values
            }),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <b>EV Charging Demand Prediction and Usage Pattern Analysis</b>
        <br>
        Machine Learning | Data Analytics | SHAP | LIME | Streamlit
        <br><br>
        Developed as a GUVI project
    </div>
    """,
    unsafe_allow_html=True
)

