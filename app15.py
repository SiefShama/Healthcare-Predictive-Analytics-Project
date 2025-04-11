import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os

# Set Streamlit page config
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

st.title("🩺 Diabetes Risk Prediction & Data Visualization App")

# --- Load ML model from Google Drive ---
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return None

# --- Load Dataset from Google Drive ---
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Data could not be loaded: {e}")
        return pd.DataFrame()

# Set your Google Drive mount paths
MODEL_PATH = "/mnt/data/diabetes_model.pkl"  # Change this if needed
DATA_START_PATH = "/mnt/data/diabetes_data_start.csv"
DATA_CLEANED_PATH = "/mnt/data/diabetes_data_cleaned.csv"

# Load model
model = load_model(MODEL_PATH)

# Sidebar: Dataset selection
st.sidebar.header("🔍 Data Selection")
dataset_version = st.sidebar.radio("Choose dataset version", ["Start", "Cleaned"])
data_path = DATA_START_PATH if dataset_version == "Start" else DATA_CLEANED_PATH
df = load_data(data_path)

# Sidebar: User inputs for prediction
st.sidebar.header("📋 User Input for Prediction")
def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0.0, 846.0, 79.0)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 21, 100, 33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction
if model:
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][int(prediction)]

    st.subheader("🧪 Prediction Result")
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.success(f"**Prediction**: {result}")
    st.info(f"**Confidence**: {prediction_proba:.2%}")

# Visualization Section
st.header("📊 Data Visualization")

if df is not None and not df.empty:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column Statistics & Value Counts")

    selected_col = st.selectbox("Select a column to visualize", df.columns)

    if selected_col in df.columns:
        try:
            counts = df[selected_col].value_counts().nlargest(20)
            fig = px.bar(
                x=counts.index,
                y=counts.values,
                labels={'x': selected_col, 'y': 'Count'},
                title=f'Value Counts of {selected_col}'
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.warning(f"Could not plot data for '{selected_col}': {e}")
    else:
        st.warning("Selected column not found in dataset.")
else:
    st.warning("DataFrame is empty or failed to load.")

st.markdown("---")
st.caption("Built for diabetes risk assessment and data analysis.")
