# EV Charging Demand Prediction and Usage Pattern Analysis

## 1. Project Overview

This project focuses on predicting electric vehicle (EV) charging demand and analyzing charging usage patterns using machine learning techniques. It helps understand historical charging behavior, estimate numerical demand, classify demand levels, and visualize the results through an interactive dashboard.

## 2. Project Objectives

- Analyze historical EV charging records.
- Clean and preprocess the dataset.
- Perform exploratory data analysis (EDA).
- Develop machine learning models for demand prediction.
- Classify charging demand into Normal Demand and High Demand.
- Compare different machine learning algorithms.
- Perform time-series forecasting on historical demand.
- Explain model predictions using SHAP and LIME.
- Build an interactive Streamlit dashboard.

## 3. Dataset

The project uses EV charging records containing information about users, chargers, locations, charger types, charging dates, charging times, duration, and demand.

Dataset files:
- `data/raw/ChargingRecords.csv`
- `data/processed/ChargingRecords_cleaned.csv`

## 4. Data Preprocessing

The following preprocessing tasks were performed:

- Loaded and inspected the dataset.
- Checked missing values and duplicate records.
- Converted date and time columns into appropriate datetime formats.
- Identified and removed invalid charging-duration records.
- Prepared the cleaned dataset for analysis and model training.

## 5. Exploratory Data Analysis

Exploratory data analysis was performed to understand charging demand and usage patterns.

- Analyzed charging records and numerical features.
- Examined demand distribution and charging duration.
- Explored usage patterns across different locations and charger categories.
- Created visualizations to understand relationships and trends in the data.

## 6. Regression Modeling

Regression models were developed to predict numerical EV charging demand.

Algorithms evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Models were compared using appropriate regression evaluation metrics, and the selected Random Forest Regressor was saved for dashboard predictions.

## 7. Classification Modeling

Classification models were developed to categorize charging demand into Normal Demand and High Demand.

Algorithms evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (LinearSVC)

The classification target was created using a demand threshold based on the dataset median. Models were evaluated and compared using classification metrics. A tuned Random Forest Classifier was saved for dashboard predictions.

## 8. Time-Series Forecasting

Time-series analysis was performed using historical daily average demand.

- Prepared daily demand data.
- Split the data chronologically into training and testing sets.
- Implemented Moving Average forecasting.
- Implemented Simple Exponential Smoothing.
- Compared forecasting performance using evaluation metrics.

## 9. Model Explainability

SHAP and LIME were implemented to improve the interpretability of machine learning predictions.

- Generated SHAP feature-importance visualizations.
- Created SHAP summary and beeswarm plots.
- Used LIME to explain individual predictions.
- Saved explanation visualizations for use in the dashboard.

## 10. Streamlit Dashboard

An interactive Streamlit dashboard was developed to present the project results.

Dashboard features:
- Dataset overview and key statistics.
- Interactive data filters.
- Charging demand and usage visualizations.
- Numerical demand prediction using regression.
- Demand-level classification.
- Machine learning model comparison charts.
- SHAP and LIME explanation visualizations.
- Filtered dataset preview.

## 11. Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Visualization:** Matplotlib, Seaborn
- **Model Explainability:** SHAP, LIME
- **Time-Series Analysis:** Statsmodels
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook, Visual Studio Code

## 12. Project Structure

```text
EV_Charging_Demand/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── classification/
│   └── regression/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_regression.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_time_series.ipynb
├── reports/
│   ├── figures/
│   └── time_series_results.csv
├── src/
│   ├── classification.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── regression.py
├── README.md
└── requirements.txt
```

## 13. How to Run the Project

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open a terminal in the project root directory.

3. Run the Streamlit dashboard:

   ```bash
   python -m streamlit run dashboard/app.py
   ```

## 14. Conclusion

This project combines data preprocessing, exploratory analysis, regression, classification, time-series forecasting, and model explainability to analyze EV charging demand. The Streamlit dashboard brings these components together in an interactive application for exploring charging patterns and viewing model predictions.