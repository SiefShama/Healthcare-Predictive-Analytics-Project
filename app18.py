import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os
import requests
import joblib
from io import BytesIO

# ------------------ Caching Functions ------------------
@st.cache_data
def load_data(dataset_version):
    file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Start.csv') if dataset_version == 'Start' else os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Mil2.csv')
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return pd.DataFrame()

@st.cache_resource(show_spinner="Loading model from Google Drive...")
def load_model_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        return None

# ------------------ Plotting Section ------------------
def plotting_section():
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
        fig = go.Figure(data=[go.Bar(x=count_data[selected_col], y=count_data["Count"], marker=dict(color=count_data["Count"], colorscale="RdBu"))])
        fig.update_layout(title=f"Distribution of {selected_col}", xaxis_title=selected_col, yaxis_title="Count")
        st.plotly_chart(fig)

    elif plot_type == "Numerical":
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

# ------------------ Prediction Section ------------------
def prediction_section():
    st.title("🩺 Diabetes Risk Prediction Form")
    st.markdown("Fill in the details below to record your health indicators.")

    model_drive_ids = {
        "pipe_DD": "1-xCpVwzAhJhKuU0zgch_x5aDvEvE4B0D",
        "gnb_DD": "1-pnjrekA5J-tD3sUUxFqoNSiwpO-zqhm",
        "svc_DD": "1-nI_HOuD7JYYj5mQfe0ALuPOZfMJuK_l",
        "knn_DD": "1-kgfXJDdFiwD6jnSz5Zk-4luzE6UY6Z3",
        "tree_DD": "1-jkof1CesquCSRHcrdDVwSVxiH2GU_a_",
        "rf_DD": "1-x_hd7qXWGBVjlhMV9ZkTK3p-A1R2sMH",
        "xgb_DD": "101OGOHoLmxWm1rnwgO3Dqq6Q6UVGTyuW",
        "mlp_DD": "100o9SlDylmRe3wnYearEmo1f4p6TA3Ni",
        "logr_DD": "1-zgV0vl8g1qWsQkrqB7nzZS_a_sXnvev"
    }

    df_models = pd.DataFrame({
        "Model": list(model_drive_ids.keys()),
        "Train Acc": [0.7895, 0.7875, 0.8002, 0.8327, 0.9825, 0.9825, 0.8128, 0.8079, 0.7897],
        "Test Acc": [0.7866, 0.7824, 0.7957, 0.7682, 0.6926, 0.7774, 0.8027, 0.8026, 0.7861],
        "R²": [0.1022, 0.0845, 0.1402, 0.0247, -0.2933, 0.0631, 0.1699, 0.1694, 0.1001],
    })
    model_data = {
    "Model": list(model_drive_ids.keys()),
    "Train Acc": [0.7895, 0.7875, 0.8002, 0.8327, 0.9825, 0.9825, 0.8128, 0.8079, 0.7897],
    "Test Acc": [0.7866, 0.7824, 0.7957, 0.7682, 0.6926, 0.7774, 0.8027, 0.8026, 0.7861],
    "R²": [0.1022, 0.0845, 0.1402, 0.0247, -0.2933, 0.0631, 0.1699, 0.1694, 0.1001],
    "Notes": [
        "Good generalization. Slight drop from train to test.",
        "Balanced performance.",
        "Strong performance.",
        "Overfitting suspected.",
        "Severe overfitting.",
        "High training score, lower generalization.",
        "Best generalization.",
        "Similar to XGB. Stable and well-balanced.",
        "Good generalization, comparable to pipe_DD."
    ]
}
    
    # Summary
    st.markdown("""
    🏆 **Top Performers**
    - XGB & MLP: Excellent balance of accuracy and generalization.
    - Logistic Regression & Pipeline: Solid results.
    - Tree models: Watch for overfitting.
    """, unsafe_allow_html=True)

    st.dataframe(df_models, use_container_width=True)
    model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")

    with st.form("diabetes_form"):
        st.subheader("Health Information")
        col1, col2 = st.columns(2)
        with col1:
            hb = st.selectbox("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cholesterol = st.selectbox("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=70)
            heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_activity = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phys_health = st.slider("Poor Physical Health Days (Last 30 Days)", 0, 30)
            gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            age = st.number_input("Age", min_value=1, max_value=120)
        with col2:
            stroke = st.radio("Stroke History", [0, 1])
            gen_health = st.selectbox("General Health", [1, 2, 3, 4, 5], format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
            chol_check = st.radio("Cholesterol Checked", [0, 1])
            smoker = st.radio("Smoker", [0, 1])
            fruits = st.radio("Consumes Fruits", [0, 1])
            veggies = st.radio("Consumes Vegetables", [0, 1])
            alcohol = st.radio("Heavy Alcohol Use", [0, 1])
            ment_health = st.slider("Poor Mental Health Days", 0, 30)
            diff_walk = st.radio("Difficulty Walking", [0, 1])
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.subheader("📋 Submitted Data")
        user_data = {
            "HB": hb, "Cholesterol": cholesterol, "BMI": bmi, "Heart_Disease": heart_disease,
            "PhysActivity": phys_activity, "PhysHlth": phys_health, "Gender": gender, "Age": age,
            "Stroke": stroke, "GenHlth": gen_health, "CholCheck": chol_check, "Smoker": smoker,
            "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": alcohol, "MentHlth": ment_health, "DiffWalk": diff_walk
        }
        df = pd.DataFrame([user_data])
        st.dataframe(df)

        if model:
            prediction = model.predict(df)[0]
            result = "🟢 Likely Healthy" if prediction == 0 else "🔴 Likely Diabetic"
            st.success(f"🧠 Model Prediction: {result}")

# ------------------ Main App Entry ------------------
def main():
    st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
    st.sidebar.title("🔎 Select Mode")
    mode = st.sidebar.radio("Choose a view:", ["📊 Visualizations", "🧠 Prediction"])

    if mode == "📊 Visualizations":
        plotting_section()
    elif mode == "🧠 Prediction":
        prediction_section()

if __name__ == "__main__":
    main()
