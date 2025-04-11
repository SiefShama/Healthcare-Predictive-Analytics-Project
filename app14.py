import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from io import BytesIO
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.set_page_config(page_title="Diabetes Risk & Insights", layout="wide")

# Cached data loading
@st.cache_data
def load_data(dataset_version):
    file_name = 'Diabetic_DB_Start.csv' if dataset_version == 'Start' else 'Diabetic_DB_Mil2.csv'
    file_path = os.path.join(os.getcwd(), 'data', file_name)
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return pd.DataFrame()

# Cached model loading from Google Drive
@st.cache_resource(show_spinner="Loading model from Google Drive...")
def load_model_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model. See details below.")
        st.exception(e)
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the app mode", ["Prediction", "Visualization"])

if app_mode == "Prediction":
    st.title("🩺 Diabetes Risk Prediction")

    # Model metadata
    model_options = {
        "Logistic Regression (88.5%)": "1McaxmAiy-r9ZCVtBBRAO_9h6s-yVl5yq",
        "Random Forest (89.9%)": "1TLoNEuDk3nPUrNOkAX_-8WhrkDRONebm",
        "XGBoost (91.3%)": "1l7W6VRQW3gNlJTXCoiDZgHgiD7P93bsw"
    }

    model_choice = st.selectbox("Select a model to use", list(model_options.keys()))
    model = load_model_from_drive(model_options[model_choice])

    if model:
        st.subheader("Enter patient information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 100, 30)
            hypertension = st.selectbox("Hypertension", [0, 1])
            bmi = st.slider("BMI", 10.0, 60.0, 25.0)
        with col2:
            heart_disease = st.selectbox("Heart Disease", [0, 1])
            HbA1c_level = st.slider("HbA1c Level", 3.0, 15.0, 5.0)
            avg_glucose_level = st.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "not current", "ever", "No Info"])

        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        smoking_map = {"never": 0, "current": 1, "former": 2, "not current": 3, "ever": 4, "No Info": 5}

        if st.button("Predict"):
            input_data = np.array([[
                age,
                hypertension,
                heart_disease,
                bmi,
                HbA1c_level,
                avg_glucose_level,
                gender_map[gender],
                smoking_map[smoking_history]
            ]])

            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]

                if prediction == 1:
                    st.error(f"🔴 High Diabetes Risk! Probability: {proba:.2%}")
                else:
                    st.success(f"🟢 Low Diabetes Risk. Probability: {proba:.2%}")
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)

elif app_mode == "Visualization":
    st.title("📊 Diabetes Dataset Visualization")

    dataset_version = st.selectbox("Select Dataset Version", ["Start", "Cleaned"])
    df = load_data(dataset_version)

    if not df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            plot_type = st.radio("Select plot type", ["Categorical", "Numerical"])

            if plot_type == "Categorical":
                selected_col = st.selectbox("Select categorical column", df.select_dtypes(include='object').columns)
                counts = df[selected_col].value_counts()
                fig = go.Figure(data=[
                    go.Bar(x=counts.index, y=counts.values)
                ])
                fig.update_layout(title=f"{selected_col} Distribution")
                st.plotly_chart(fig, use_container_width=True)

            else:
                selected_col = st.selectbox("Select numerical column", df.select_dtypes(include=np.number).columns)
                values = df[selected_col].dropna()
                kde = gaussian_kde(values)
                x_vals = np.linspace(values.min(), values.max(), 200)
                y_vals = kde(x_vals)
                fig = go.Figure(data=[
                    go.Scatter(x=x_vals, y=y_vals, fill='tozeroy', name='Density')
                ])
                fig.update_layout(title=f"{selected_col} Distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for visualization.")
