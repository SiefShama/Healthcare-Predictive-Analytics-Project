import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os
import joblib
import requests
from io import BytesIO

# ---------- Caching Functions ----------
@st.cache_data
def load_data(dataset_version):
    if dataset_version == 'Start':
        file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Start.csv')
    else:
        file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Mil2.csv')

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="Loading model...")
def load_model_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model")
        st.exception(e)
        return None

# ---------- Model Links ----------
model_drive_ids = {
    "xgb_DD": "101OGOHoLmxWm1rnwgO3Dqq6Q6UVGTyuW",
}

# ---------- Sidebar Navigation ----------
st.set_page_config(page_title="Diabetes App", layout="wide")
st.sidebar.title("📌 Choose Action")
app_mode = st.sidebar.radio("Go to", ["🔍 Visualizations", "🩺 Prediction"])

# ---------- Visualizations ----------
if app_mode == "🔍 Visualizations":
    st.title("📊 Diabetes Dataset Visualizations")
    dataset_version = st.sidebar.selectbox("Choose dataset version:", ["Start", "Cleaned"])
    Diabetic_DB = load_data(dataset_version)

    categorical_columns = [
        "Diabetes_State", "PhysHlth", "Gender", "Age", "Stroke", "GenHlth",
        "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"
    ]
    numerical_columns = ["HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity", "MentHlth"]

    plot_type = st.sidebar.selectbox("Choose plot type:", ["Categorical", "Numerical"])

    if plot_type == "Categorical":
        selected_col = st.selectbox("Select a categorical column:", categorical_columns)
        count_data = Diabetic_DB[selected_col].value_counts().reset_index()
        count_data.columns = [selected_col, "Count"]

        fig = go.Figure(data=[go.Bar(
            x=count_data[selected_col],
            y=count_data["Count"],
            marker=dict(color=count_data["Count"], colorscale="RdBu"),
        )])
        fig.update_layout(title=f"Distribution of {selected_col}", xaxis_title=selected_col, yaxis_title="Count")
        st.plotly_chart(fig)

    else:
        selected_col = st.selectbox("Select a numerical column:", numerical_columns)
        data = Diabetic_DB[selected_col].dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Histogram", marker_color='royalblue', opacity=0.7, histnorm='probability density'))
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='KDE', line=dict(color='crimson')))
        fig.update_layout(title=f"Histogram with KDE for {selected_col}", xaxis_title=selected_col, yaxis_title="Density", barmode='overlay')
        st.plotly_chart(fig)

# ---------- Prediction ----------
elif app_mode == "🩺 Prediction":
    st.title("🩺 Diabetes Risk Input Form")
    st.markdown("Fill in the form to get a diabetes risk prediction.")

    model_id = model_drive_ids["xgb_DD"]
    model = load_model_from_drive(model_id)

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)

        with col1:
            hb = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cholesterol = st.selectbox("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
            bmi = st.number_input("BMI", 10, 70)
            heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_activity = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_health = st.slider("Poor Physical Health Days (30d)", 0, 30)
            gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age", 1, 120)

        with col2:
            stroke = st.radio("Stroke History", [0, 1], format_func=lambda x: "Yes" if x else "No")
            gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
            chol_check = st.radio("Cholesterol Checked", [0, 1], format_func=lambda x: "Yes" if x else "No")
            smoker = st.radio("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
            fruits = st.radio("Eats Fruits", [0, 1], format_func=lambda x: "Yes" if x else "No")
            veggies = st.radio("Eats Veggies", [0, 1], format_func=lambda x: "Yes" if x else "No")
            alcohol = st.radio("Heavy Alcohol Use", [0, 1], format_func=lambda x: "Yes" if x else "No")
            ment_health = st.slider("Poor Mental Health Days (30d)", 0, 30)
            diff_walk = st.radio("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("Submit")

    if submitted and model:
        st.markdown("---")
        st.subheader("📋 Submitted Information")
        user_data = {
            "HB": hb, "Cholesterol": cholesterol, "BMI": bmi, "Heart_Disease": heart_disease,
            "PhysActivity": phys_activity, "PhysHlth": phys_health, "Gender": gender, "Age": age,
            "Stroke": stroke, "GenHlth": gen_health, "CholCheck": chol_check, "Smoker": smoker,
            "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": alcohol,
            "MentHlth": ment_health, "DiffWalk": diff_walk
        }

        input_df = pd.DataFrame([user_data])
        st.dataframe(input_df)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of being diabetic

        st.markdown("### 🧠 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ You are likely **Diabetic**")
        else:
            st.success(f"✅ You are likely **Not Diabetic**")

        st.info(f"📊 **Probability of being diabetic:** `{probability:.2%}`")
