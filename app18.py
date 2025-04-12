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
    
    # Define the dataset schema
    data1 = {
        "Column Name": [
            "Diabetes_State", "HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity",
            "PhysHlth", "Gender", "Age", "Stroke", "GenHlth", "CholCheck", "Smoker",
            "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "DiffWalk", "ID", "Race", "Weight"
        ],
        "Data Type": [
            "int64", "float64", "float64", "float64", "float64", "float64",
            "int64", "float64", "int64", "float64", "float64", "float64", "float64",
            "float64", "float64", "float64", "float64", "float64", "int64", "object", "object"
        ],
        "Description": [
            "Indicates if the individual has diabetes (`1` = Yes, `0` = No).",
            "High blood pressure level.",
            "Total cholesterol level.",
            "Body Mass Index (BMI).",
            "Indicates history of heart disease or heart attack.",
            "Whether the individual exercises regularly (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor physical health.",
            "Gender (`1` = Male, `0` = Female).",
            "Age group classification.",
            "Whether the individual has had a stroke (`1` = Yes, `0` = No).",
            "Self-reported general health (`1` = Excellent, ... , `5` = Poor).",
            "Whether cholesterol was checked in the past 5 years (`1` = Yes, `0` = No).",
            "Whether the individual is a smoker (`1` = Yes, `0` = No).",
            "Whether the individual consumes fruits regularly (`1` = Yes, `0` = No).",
            "Whether the individual consumes vegetables regularly (`1` = Yes, `0` = No).",
            "Heavy alcohol consumption (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor mental health.",
            "Difficulty walking or climbing stairs (`1` = Yes, `0` = No).",
            "Unique patient identifier.",
            "Patient's racial background.",
            "Weight of the patient (missing for most records)."
        ],
        "Potential Use in Analysis": [
            "Target variable for diabetes prediction models.",
            "Risk factor for diabetes and heart disease.",
            "High levels may indicate risk of diabetes or cardiovascular disease.",
            "Used to assess obesity, a risk factor for diabetes.",
            "Useful for assessing comorbidities with diabetes.",
            "Physical activity helps in managing diabetes.",
            "Indicator of overall health and chronic conditions.",
            "Used for demographic analysis in health studies.",
            "Age is a major risk factor for diabetes.",
            "Stroke risk increases with diabetes.",
            "Subjective health assessment for predictive models.",
            "Preventive health measure for cardiovascular risks.",
            "Smoking is a risk factor for diabetes complications.",
            "Affects diet-related diabetes risk.",
            "Indicator of a healthy diet.",
            "Excessive drinking can increase diabetes risk.",
            "Mental health impacts overall well-being.",
            "Indicator of mobility issues related to diabetes.",
            "Used for tracking individual records.",
            "Useful for studying disparities in diabetes prevalence.",
            "Can be used for BMI calculations if data is available."
        ]
    }
    

    # Define the dataset schema
    data2 = {
        "Column Name": [
            "Diabetes_State", "HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity",
            "PhysHlth", "Gender", "Age", "Stroke", "GenHlth", "CholCheck", "Smoker",
            "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "DiffWalk"
        ],
        "Data Type": [
            "int64", "int64", "int64", "int64", "int64", "int64",
            "int64", "int64", "int64", "int64", "int64", "int64", "int64",
            "int64", "int64", "int64", "int64", "int64"
        ],
        "Description": [
            "Indicates if the individual has diabetes (`1` = Yes, `0` = No).",
            "High blood pressure level.",
            "Total cholesterol level.",
            "Body Mass Index (BMI).",
            "Indicates history of heart disease or heart attack.",
            "Whether the individual exercises regularly (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor physical health.",
            "Gender (`1` = Male, `0` = Female).",
            "Age group classification.",
            "Whether the individual has had a stroke (`1` = Yes, `0` = No).",
            "Self-reported general health (`1` = Excellent, ... , `5` = Poor).",
            "Whether cholesterol was checked in the past 5 years (`1` = Yes, `0` = No).",
            "Whether the individual is a smoker (`1` = Yes, `0` = No).",
            "Whether the individual consumes fruits regularly (`1` = Yes, `0` = No).",
            "Whether the individual consumes vegetables regularly (`1` = Yes, `0` = No).",
            "Heavy alcohol consumption (`1` = Yes, `0` = No).",
            "Number of days in the past month with poor mental health.",
            "Difficulty walking or climbing stairs (`1` = Yes, `0` = No)."
        ],
        "Potential Use in Analysis": [
            "Target variable for diabetes prediction models.",
            "Risk factor for diabetes and heart disease.",
            "High levels may indicate risk of diabetes or cardiovascular disease.",
            "Used to assess obesity, a risk factor for diabetes.",
            "Useful for assessing comorbidities with diabetes.",
            "Physical activity helps in managing diabetes.",
            "Indicator of overall health and chronic conditions.",
            "Used for demographic analysis in health studies.",
            "Age is a major risk factor for diabetes.",
            "Stroke risk increases with diabetes.",
            "Subjective health assessment for predictive models.",
            "Preventive health measure for cardiovascular risks.",
            "Smoking is a risk factor for diabetes complications.",
            "Affects diet-related diabetes risk.",
            "Indicator of a healthy diet.",
            "Excessive drinking can increase diabetes risk.",
            "Mental health impacts overall well-being.",
            "Indicator of mobility issues related to diabetes."
        ]
    }

    # Convert to DataFrame
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Display with Streamlit
    if dataset_version == "Start":
        st.title("📊 Diabetic DB - Dataset Schema (Start Version)")
        st.dataframe(df1)
    else:
        st.title("📋 Diabetic DB - Dataset Schema (CLeaned Version)")
        st.dataframe(df2)
    
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


    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    
    model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")

    with st.form("diabetes_form"):
        st.subheader("Health Information")
        col1, col2 = st.columns(2)
        with col1:
            hb = st.radio("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cholesterol = st.radio("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
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
        st.markdown("---")
        st.subheader("📋 Submitted Information")

        user_data = {
        "HB": hb,
        "Cholesterol": cholesterol,
        "BMI": bmi,
        "Heart_Disease": heart_disease,
        "PhysActivity": phys_activity,
        "PhysHlth": phys_health,
        "Gender": gender,
        "Age": age,
        "Stroke": stroke,
        "GenHlth": gen_health,
        "CholCheck": chol_check,
        "Smoker": smoker,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "MentHlth": ment_health,
        "DiffWalk": diff_walk
        }

        df = pd.DataFrame([user_data])
        st.dataframe(df)
        st.success("✅ Your input has been recorded!")


        if model:
            prediction = model.predict(df)[0]
            result = "🟢 Likely Healthy" if prediction == 0 else "🔴 Likely Diabetic"
            st.success(f"🧠 Model Prediction: {result}")


            
        user_input = [[
            hb, cholesterol, bmi, heart_disease, phys_activity,
            phys_health, gender, age, stroke, gen_health,
            chol_check, smoker, fruits, veggies, alcohol,
            ment_health, diff_walk
        ]]


        model_id = model_drive_ids.get(model_choice)

        if model_id:
            model = load_model_from_drive(model_id)
            if model:
                try:
                    prediction = model.predict(user_input)[0]
                    pred_label = "Diabetic" if prediction == 1 else "Non-Diabetic"
                    st.success(f"🧾 **Prediction Result:** {pred_label}")
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(user_input)[0][1]
                        st.info(f"📊 Probability of being diabetic: **{prob:.2%}**")
                except Exception as e:
                    st.error("❌ Prediction failed. See error below.")
                    st.exception(e)
            else:
                st.error("❌ Model could not be loaded.")
        else:
            st.error("❌ Selected model ID is missing.")




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
