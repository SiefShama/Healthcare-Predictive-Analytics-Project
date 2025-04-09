import streamlit as st
import pandas as pd
import os
import joblib
import gdown

# --- File Setup ---
DATA_PATH = 'data'
MODEL_PATH = 'models'
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# --- Model Files on Google Drive ---
model_files = {
    "pipe_DD": "1-xCpVwzAhJhKuU0zgch_x5aDvEvE4B0D",
    "regressor_DD": "1-tmgxZHBXoMjnyx8iHdzBORbHS43bCVj",
    "gnb_DD": "1-pnjrekA5J-tD3sUUxFqoNSiwpO-zqhm",
    "svc_DD": "1-nI_HOuD7JYYj5mQfe0ALuPOZfMJuK_l",
    "knn_DD": "1-kgfXJDdFiwD6jnSz5Zk-4luzE6UY6Z3",
    "tree_DD": "1-jkof1CesquCSRHcrdDVwSVxiH2GU_a_",
    "rf_DD": "1-x_hd7qXWGBVjlhMV9ZkTK3p-A1R2sMH",
    "xgb_DD": "101OGOHoLmxWm1rnwgO3Dqq6Q6UVGTyuW",
    "mlp_DD": "100o9SlDylmRe3wnYearEmo1f4p6TA3Ni",
    "logr_DD": "1-zgV0vl8g1qWsQkrqB7nzZS_a_sXnvev",
    "treer_DD": "1-zct0945l0EESuVkW51f60oFQXJC2AIt",
    "rfr_DD": "102J_kkpxhcdpbGOJzSXBcH5-g-RHVhXj",
    "xgbr_DD": "10A2tkOSszRryIa1PDbxyGefVj2w1-7aE",
    "mlpr_DD": "106ZKudJcHg--Q1OeFfQ-g1CF4Ps_KGpD",
    "knnr_DD": "102fam3U0c63bHIypOD6Wh_84obw6MPjg"
}

# --- Clear gdown Cache ---
def clear_cache():
    cache_dir = os.path.expanduser('~/.cache/gdown')
    if os.path.exists(cache_dir):
        os.system(f'rm -rf {cache_dir}')

# --- Download Models if Not Already ---
def download_models():
    for name, file_id in model_files.items():
        model_path = os.path.join(MODEL_PATH, f"{name}.pkl")
        if not os.path.exists(model_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)

# --- Load Models ---
def load_models():
    models = {}
    for name in model_files.keys():
        path = os.path.join(MODEL_PATH, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

# --- Dataset Loader ---
def load_data(dataset_version):
    file = "Diabetic_DB_Start.csv" if dataset_version == "Start" else "Diabetic_DB_Mil2.csv"
    return pd.read_csv(os.path.join(DATA_PATH, file))

# --- UI Layout ---
st.title("Diabetes Prediction & Visualization App")

# --- Dataset Selection ---
dataset_version = st.sidebar.selectbox("Select Dataset Version", options=["Start", "Cleaned"])
data = load_data(dataset_version)
st.sidebar.success(f"Loaded dataset: {dataset_version}")

# --- Mode Toggle ---
mode = st.sidebar.radio("Choose View", ["🤖 ML Predictions"])

# --- Run Download and Load Models ---
clear_cache()  # Clear cache to avoid any old cookie issues
download_models()
loaded_models_DD = load_models()

# --- Display Content Based on Mode ---
if mode == "🤖 ML Predictions":
    st.header("🤖 ML Prediction Form")

    # --- Model Selection ---
    model_name = st.selectbox("Choose a model", list(model_files.keys()))
    model = loaded_models_DD.get(model_name)

    if not model:
        st.error("Model not loaded. Check your Drive links.")
    else:
        # --- User Form ---
        st.title('Health Data Form')

        diabetes_state = st.selectbox("Diabetic?", ["Yes", "No"])
        hb = st.selectbox("Healthy behavior?", ["Yes", "No"])
        cholesterol = st.selectbox("High cholesterol?", ["Yes", "No"])
        bmi = st.number_input("BMI", 13.0, 41.0, step=0.1)
        heart_disease = st.selectbox("Heart disease?", ["Yes", "No"])
        phys_activity = st.checkbox("Physical activity")
        phys_health = st.number_input("Days physical health not good", 0, 12, step=1)
        gender = st.radio("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 95, step=1)
        stroke = st.selectbox("Stroke?", ["Yes", "No"])
        gen_health = st.radio("General health (1-5)", [1, 2, 3, 4, 5])
        chol_check = st.checkbox("Had cholesterol check in last 5 years?")
        smoker = st.checkbox("Smoker?")
        fruits = st.checkbox("Eats fruits regularly?")
        veggies = st.checkbox("Eats vegetables regularly?")
        heavy_alcohol = st.checkbox("Heavy alcohol consumption?")
        mental_health = st.number_input("Days mental health not good", 0, 5, step=1)
        diff_walk = st.checkbox("Difficulty walking?")

        # --- Prediction ---
        if st.button("Predict"):
            input_data = pd.DataFrame([[ 
                1 if diabetes_state == "Yes" else 0,
                1 if hb == "Yes" else 0,
                1 if cholesterol == "Yes" else 0,
                bmi,
                1 if heart_disease == "Yes" else 0,
                1 if phys_activity else 0,
                phys_health,
                1 if gender == "Male" else 0,
                age,
                1 if stroke == "Yes" else 0,
                gen_health,
                1 if chol_check else 0,
                1 if smoker else 0,
                1 if fruits else 0,
                1 if veggies else 0,
                1 if heavy_alcohol else 0,
                mental_health,
                1 if diff_walk else 0
            ]], columns=[
                "Diabetes_binary", "HB", "Chol_check", "BMI", "HeartDiseaseorAttack",
                "PhysActivity", "PhysHlth", "Sex", "Age", "Stroke", "GenHlth",
                "HighChol", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump",
                "MentHlth", "DiffWalk"
            ])

            try:
                prediction = model.predict(input_data)[0]
                st.success(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
