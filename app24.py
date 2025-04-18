import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde
import os
import requests
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import seaborn as sns
from sklearn.model_selection import train_test_split  # Split dataset into training and testing sets
from sklearn.metrics import auc





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
        st.title("📋 Dataset Schema (Start Version)")
        st.dataframe(df1)
   
    else:
        st.title("📋 Dataset Schema (CLeaned Version)")
        st.dataframe(df2)
    
    categorical_columns = [
        "Diabetes_State", "Gender", "Stroke", "GenHlth",
        "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk",
        "HB","Cholesterol", "Heart_Disease", "PhysActivity"
    ]
    numerical_columns = [ "BMI", "MentHlth", "Age", "PhysHlth"]

    plot_type = st.sidebar.selectbox("Choose plot type:", ["Categorical", "Numerical", "General"])

    if plot_type == "Categorical":
        selected_col = st.selectbox("Select a categorical column:", categorical_columns)
        count_data = Diabetic_DB[selected_col].value_counts().reset_index()
        count_data.columns = [selected_col, "Count"]
        fig = go.Figure(data=[go.Bar(x=count_data[selected_col], y=count_data["Count"], marker=dict(color=count_data["Count"], colorscale="RdBu"))])
        fig.update_layout(title=f"Distribution of {selected_col}", xaxis_title=selected_col, yaxis_title="Count")
        st.plotly_chart(fig)
        if dataset_version == "Start":
            
            st.markdown("### 🔍 **Overall Dataset Insights**")
            st.markdown("""

            Imbalances: Many target variables (e.g., Stroke, Heart_Disease, Diabetes_State) show strong class imbalance, which could affect predictive modeling and need handling during training.
            Balanced Features: Variables like Gender, Smoker, HB, and Cholesterol are more evenly distributed, contributing well to feature diversity.
            Lifestyle Factors: Variables like PhysActivity, Fruits, Veggies, and HvyAlcoholConsum suggest respondents report relatively healthy behaviors.

            """, unsafe_allow_html=True)
            if selected_col =="Diabetes_State":
                st.markdown("#### 🔷 **Distribution of Diabetes_State**")
                st.markdown("""
                0: Majority (~220,000+ samples) do not have diabetes.
                1: Smaller portion (~100,000 samples) have diabetes.
                Comment: There's class imbalance here. If this is the target variable for a model, class balancing techniques might be needed (e.g., SMOTE, class weights).

                """, unsafe_allow_html=True)
                
            elif selected_col == "Gender":
                st.markdown("#### 🔷 **Distribution of Gender**")
                st.markdown("""
                0 and 1 (probably Male and Female or vice versa): Both classes are well represented.
                Comment: The gender variable appears balanced, which is good for modeling to avoid gender bias.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Stroke":
                st.markdown("#### 🔷 **Distribution of Stroke**")
                st.markdown("""
                0: Vast majority (~240,000) did not experience stroke.
                1: Very small portion (~10,000) had a stroke.
                Comment: Very imbalanced, potential challenge for binary classification tasks. Important for precision-recall metrics over accuracy.

                """, unsafe_allow_html=True)
                
            elif selected_col == "GenHlth":
                st.markdown("#### 🔷 **Distribution of GenHlth (General Health)**")
                st.markdown("""
                Values from 1 (Excellent) to 5 (Poor):
                2 and 3 (Good to Very Good) are most common.
                Value 5 (Poor) is the least.
                Comment: Shows a realistic distribution of self-rated health with a skew towards better health.

                """, unsafe_allow_html=True)
                
            elif selected_col == "CholCheck":
                st.markdown("#### 🔷 **Distribution of CholCheck (Cholesterol Check)**")
                st.markdown("""
                1 (Yes) dominates — most people had their cholesterol checked.
                0 (No) is very low.
                Comment: Indicates good health awareness in the dataset sample.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Smoker":
                st.markdown("#### 🔷 **Distribution of Smoker**")
                st.markdown("""
                Slightly more non-smokers than smokers.
                Comment: The distribution is relatively balanced; might serve well in exploring lifestyle impacts on health.

                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Fruits":
                st.markdown("#### 🔷 **Distribution of Fruits**")
                st.markdown("""
                Majority consume fruits (1).
                Minority do not (0).
                Comment: Positive lifestyle behavior; this can serve as a health feature indicating awareness.

                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Veggies":
                st.markdown("#### 🔷 **Distribution of Veggies**")
                st.markdown("""
                Similar to Fruits — majority consume vegetables.
                Comment: Another lifestyle indicator with more healthy responders.
                """, unsafe_allow_html=True)
                
            elif selected_col == "HvyAlcoholConsump":
                st.markdown("#### 🔷 **Distribution of HvyAlcoholConsum (Heavy Alcohol Consumption)**")
                st.markdown("""
                0 (No): Strong majority.
                1 (Yes): Very few.
                Comment: Imbalance again; very few heavy drinkers, so feature may have limited discriminative power in models unless highly correlated.

                """, unsafe_allow_html=True)
                  
            elif selected_col == "DiffWalk":
                st.markdown("#### 🔷 **Distribution of DiffWalk (Difficulty Walking)**")
                st.markdown("""
                0: Majority have no difficulty.
                1: Minority face difficulty.
                Comment: May correlate with age, disability, or chronic disease indicators — useful for deeper analysis.
                """, unsafe_allow_html=True)
            
            elif selected_col == "HB":
                st.markdown("#### 🔷 **Distribution of HB (High Blood Pressure)**")
                st.markdown("""
                Reasonably balanced:
                Many have high blood pressure (1).
                Many don’t (0).
                Comment: Potentially a strong predictor of both heart disease and diabetes.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Cholesterol":
                st.markdown("#### 🔷 **Distribution of Cholesterol**")
                st.markdown("""
                Fairly balanced:
                Slightly more individuals without high cholesterol (0).
                Many with high cholesterol (1).
                Comment: Useful for cardiovascular risk models.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Heart_Disease":
                st.markdown("#### 🔷 **Distribution of Heart_Disease**")
                st.markdown("""
                0: Vast majority without heart disease.
                1: Small minority with heart disease.
                Comment: Another imbalanced target if used in prediction. Oversampling may be needed.
                """, unsafe_allow_html=True)
                
            elif selected_col == "PhysActivity":
                st.markdown("#### 🔷 **Distribution of PhysActivity (Physical Activity)**")
                st.markdown("""
                Majority engage in physical activity (1).
                Minority do not (0).
                Comment: Important lifestyle variable; inverse relationship with many chronic diseases.
                """, unsafe_allow_html=True)
                  

        else:
            st.markdown("### 🔍 **Overall Dataset Insights**")
            st.markdown("""
            The dataset reflects a generally healthy population, with most individuals free from diabetes, heart disease, or stroke. However, a notable minority has these conditions, highlighting key risk groups.
            Lifestyle habits are mixed — many show poor physical activity, low fruit/vegetable intake, and some smoke or drink heavily, all contributing to health risks.
            Health awareness is moderate, with many having checked their cholesterol, but gaps remain. General health perception is mostly positive, though some report poor health and mobility issues, possibly linked to aging or chronic illness.
            These insights emphasize the need for targeted health promotion, early screening, and lifestyle interventions.

            """, unsafe_allow_html=True)
            

            if selected_col =="Diabetes_State":
                st.markdown("#### 🔷 **Distribution of Diabetes_State**")
                st.markdown("""
                - This column represents whether a person has diabetes (1) or not (0).  
                - The bar plot likely shows that the majority of individuals do not have diabetes, with a smaller proportion being diabetic.  
                - This imbalance suggests that diabetes cases are less frequent in the dataset, which is expected in a general population dataset.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Gender":
                st.markdown("#### 🔷 **Distribution of Gender**")
                st.markdown("""
                The dataset has a mix of male (0) and female (1) respondents.
                If there's an imbalance, it could indicate sampling bias where one gender is more represented than the other.
                Understanding gender distribution helps in assessing how health conditions vary between males and females.
                
                """, unsafe_allow_html=True)
                
            elif selected_col == "Stroke":
                st.markdown("#### 🔷 **Distribution of Stroke**")
                st.markdown("""
                The majority of individuals are expected to have 0 (no stroke), while a smaller percentage has 1 (had a stroke).
                The small number of stroke cases aligns with real-world data, as strokes are less frequent in the general population.
                This variable is crucial for analyzing cardiovascular risks.
                """, unsafe_allow_html=True)
                
            elif selected_col == "GenHlth":
                st.markdown("#### 🔷 **Distribution of GenHlth (General Health)**")
                st.markdown("""
                This is likely an ordinal scale (e.g., 1 = Excellent, 5 = Poor).
                If the bar plot is skewed towards the lower values (1 or 2), it suggests that most people rate their health as good.
                A significant number of people in the higher categories (4 or 5) could indicate a population with existing health concerns.
                """, unsafe_allow_html=True)
                
            elif selected_col == "CholCheck":
                st.markdown("#### 🔷 **Distribution of CholCheck (Cholesterol Check)**")
                st.markdown("""
                This binary variable (0 = No, 1 = Yes) indicates whether individuals have checked their cholesterol levels.
                If the majority fall into 1, it suggests a health-conscious population.
                If many haven't checked (0), it might indicate a need for better health screening programs.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Smoker":
                st.markdown("#### 🔷 **Distribution of Smoker**")
                st.markdown("""
                The dataset likely shows a mix of smokers (1) and non-smokers (0).
                A high percentage of smokers suggests a public health concern, as smoking is linked to diabetes and cardiovascular diseases.
                This variable is crucial for understanding lifestyle risk factors.
                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Fruits":
                st.markdown("#### 🔷 **Distribution of Fruits**")
                st.markdown("""
                This variable is likely binary (1 = Eats fruits regularly, 0 = Does not).
                If most people fall into 0, it suggests poor dietary habits.
                Higher fruit consumption (more 1s) is associated with better overall health.
                """, unsafe_allow_html=True)
                
                
            elif selected_col == "Veggies":
                st.markdown("#### 🔷 **Distribution of Veggies**")
                st.markdown("""
                Like the fruits variable, this indicates vegetable consumption.
                If more people report 0, it suggests a lack of proper nutrition, which could be a risk factor for diabetes.
                A higher number of 1s would be a positive indicator of healthy eating habits.
                """, unsafe_allow_html=True)
                
            elif selected_col == "HvyAlcoholConsump":
                st.markdown("#### 🔷 **Distribution of HvyAlcoholConsum (Heavy Alcohol Consumption)**")
                st.markdown("""
                This variable indicates individuals who consume alcohol heavily (1) versus those who do not (0).
                If the majority are 0, it suggests that most individuals do not engage in heavy drinking.
                Higher numbers in 1 could indicate a health concern, as excessive alcohol is linked to various diseases, including diabetes and heart conditions.
                """, unsafe_allow_html=True)
                  
            elif selected_col == "DiffWalk":
                st.markdown("#### 🔷 **Distribution of DiffWalk (Difficulty Walking)**")
                st.markdown("""
                This variable indicates if a person has mobility issues (1) or not (0).
                A high count in 1 could indicate a population with chronic illnesses or aging individuals.
                Mobility issues are often associated with diabetes complications or heart diseases.
                """, unsafe_allow_html=True)
                            
            elif selected_col == "HB":
                st.markdown("#### 🔷 **Distribution of HB (High Blood Pressure)**")
                st.markdown("""
                Hemoglobin levels typically follow a normal distribution, but the histogram might show a skewed pattern due to missing values.
                A peak in a normal range (e.g., 12–17 g/dL) suggests most individuals have healthy hemoglobin levels.
                If there are outliers (extremely low or high values), it could indicate anemia or polycythemia, both of which can be linked to diabetes complications.
                """, unsafe_allow_html=True)
                
            elif selected_col == "Cholesterol":
                st.markdown("#### 🔷 **Distribution of Cholesterol**")
                st.markdown("""
                The histogram likely shows a right-skewed distribution, meaning most people have cholesterol in the normal range but some have very high values.
                High cholesterol is a major risk factor for diabetes and heart disease.
                If many individuals have high cholesterol levels, it highlights a significant health risk in the population.

                """, unsafe_allow_html=True)
                
            elif selected_col == "Heart_Disease":
                st.markdown("#### 🔷 **Distribution of Heart_Disease**")
                st.markdown("""
                The values might be binary or continuous, representing risk scores.
                If binary (0 = No, 1 = Yes), a majority 0 indicates fewer heart disease cases, while a significant number of 1s highlights a concerning trend.
                If continuous, a right-skewed distribution suggests a high-risk population.
                """, unsafe_allow_html=True)
                
            elif selected_col == "PhysActivity":
                st.markdown("#### 🔷 **Distribution of PhysActivity (Physical Activity)**")
                st.markdown("""
                This variable may show a peak at 0 (no activity) and then a gradual decrease at higher activity levels.
                A high number of 0s suggests a sedentary lifestyle, which is a risk factor for diabetes.
                If the histogram shows a spread-out distribution, it means some individuals are highly active, while others are not.
                """, unsafe_allow_html=True)
            
            
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
        if dataset_version == "Start":
            st.markdown("### 🧠 **Insights and Recommendations**")
            st.markdown("""

            1. **Right-Skewed Distributions:**
               - **MentHlth** and **PhysHlth** exhibit **strong right skew**, indicating most individuals report **no unhealthy days** in the past month.
               - These features highlight a population skewed toward better health, with a minority experiencing chronic or persistent health issues.

            2. **BMI (Body Mass Index):**
               - Displays a **realistic population distribution**, concentrated between **20–35**, with a notable **peak near 27–28**.
               - Presence of **extreme outliers (BMI > 50)** may affect model training unless addressed.
               - A majority fall in the **overweight or obese** category, aligning with common chronic disease risk trends.

            3. **Age Variable:**
               - Appears **categorical or bucketed** rather than continuous.
               - **Multimodal distribution** implies groupings like age bands (e.g., 18–24, 25–29, etc.).
               - This needs clarification as it directly affects encoding strategies in modeling.
            """, unsafe_allow_html=True)



            if selected_col =="BMI":
                st.markdown("#### 🔷 **Distribution of BMI (Body Mass Index)**")
                st.markdown("""
                Distribution: Right-skewed (positively skewed).
                Peak: Most data points are concentrated between 20 and 35 BMI, with a prominent peak around 27-28.
                Outliers: Few individuals have BMI above 50, extending even to 90+.
                Comment:
                    The distribution is realistic and typical for population-based health surveys.
                    BMI values above 30 typically indicate obesity — a major risk factor for diabetes and heart disease.
                    The presence of outliers may affect model performance if not handled (e.g., with normalization, clipping, or log transformation).

                """, unsafe_allow_html=True)
            elif selected_col == "MentHlth":
                st.markdown("#### 🔷 **Distribution of MentHlth (Number of Mentally Unhealthy Days in Last 30)**")
                st.markdown("""
                Distribution: Strong right skew.
                Peak at 0: A huge spike at 0 days (indicating no mentally unhealthy days for most respondents).
                Spread: Gradual decline up to 30 days. Minor peaks near 5, 10, and 30.
                Comment:
                    Most individuals report no or few mentally unhealthy days — common in health self-reports.
                    The spike at 30 days might indicate chronic or severe mental health issues for a small group.
                    This skewed nature may benefit from transformation or binning in modeling.

                """, unsafe_allow_html=True)
            elif selected_col == "Age":
                st.markdown("#### 🔷 **Distribution of Age**")
                st.markdown("""
                Distribution: Multimodal (multiple peaks).
                Strange Binning: Appears categorical or grouped by age brackets (e.g., 5-year intervals). Peaks at fixed intervals (e.g., 18, 25, 30... up to 80+).
                Peak: Highest concentration around early 20s, then steady peaks through older ages.
                Comment:
                    Age might be encoded as buckets or group codes rather than actual age (e.g., 1 = 18-24, 2 = 25-29, etc.).
                    Clarify if these are codes or exact ages — affects feature engineering.
                    Age plays a crucial role in health outcome prediction, but should be continuous or properly one-hot encoded if categorical.

                """, unsafe_allow_html=True)
            elif selected_col == "PhysHlth":
                st.markdown("#### 🔷 **Distribution of PhysHlth (Number of Physically Unhealthy Days in Last 30)**")
                st.markdown("""
                Distribution: Strong right skew — very similar to MentHlth.
                Peak at 0: Majority of the respondents report 0 physically unhealthy days.
                Tail: Long tail reaching to 30 with minor spikes along the way.
                Comment:
                    Shows that many people are in good physical health.
                    Spikes at values like 10 and 30 may be due to rounding or special cases (chronic illness).
                    Like MentHlth, this feature might benefit from grouping (e.g., 0 days, 1-5, 6-15, 16-30) in classification models or visualizations.
     
                """, unsafe_allow_html=True)
            
        else:
            st.markdown("### 🧠 **Insights and Recommendations**")
            st.markdown("""            
            The data shows trends typical of health-focused populations. 
            Most individuals have normal to slightly high BMI, with a significant portion potentially obese — reinforcing obesity’s link to diabetes. Physical and mental health are generally good, though a minority experience chronic issues, indicating potential risk groups. 
            The age distribution appears skewed toward older individuals, aligning with higher risks for chronic diseases like diabetes and heart disease. 
            These patterns suggest strong potential for predicting health outcomes using lifestyle and age-related features.

            """, unsafe_allow_html=True)
            
            if selected_col =="BMI":
                st.markdown("#### 🔷 **Distribution of BMI (Body Mass Index)**")
                st.markdown("""
                The BMI histogram is likely skewed toward higher values, as obesity is common in diabetes datasets.
                A peak in the 18.5–24.9 range indicates normal weight, while peaks in the 25–30+ range suggest overweight or obesity.
                If many individuals fall into BMI > 30, it reinforces the link between obesity and diabetes.
                """, unsafe_allow_html=True)
            elif selected_col == "MentHlth":
                st.markdown("#### 🔷 **Distribution of MentHlth (Number of Mentally Unhealthy Days in Last 30)**")
                st.markdown("""
                This histogram might be right-skewed, with most individuals reporting 0–5 mentally unhealthy days.
                If there is a significant number of people with 10+ mentally unhealthy days, it suggests mental health challenges in the population.
                Poor mental health is linked to chronic disease management, including diabetes.
                """, unsafe_allow_html=True)
            elif selected_col == "Age":
                st.markdown("#### 🔷 **Distribution of Age**")
                st.markdown("""
                The age column is encoded, possibly in categories (e.g., 1 = young adults, 9 = middle-aged, etc.).
                The bar chart might show peaks in certain age ranges, indicating a higher representation of certain age groups.
                If older age groups dominate, it could suggest a bias towards older populations, who are more likely to have health conditions.

                """, unsafe_allow_html=True)
            elif selected_col == "PhysHlth":
                st.markdown("#### 🔷 **Distribution of PhysHlth (Number of Physically Unhealthy Days in Last 30)**")
                st.markdown("""
                The distribution is likely skewed towards 0 (indicating good physical health for most people).
                A small number of individuals report higher values, suggesting they experienced many days of poor health.
                This variable could be important in predicting diabetes and heart disease, as prolonged poor physical health is often linked to chronic conditions.
                     
                """, unsafe_allow_html=True)

            
    elif plot_type == "General":
        ## Compute correlation matrix
        correlation_matrix = Diabetic_DB.corr()

        # Create heatmap using graph_objects
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            hoverongaps=False
        ))

        # Make it taller (height in pixels)
        fig.update_layout(
            title='Correlation Heatmap',
            xaxis_nticks=36,
            height=800  # <-- Change this value to control vertical size
        )

        
        
        # Display in Streamlit
        st.title("Correlation Heatmap of Diabetic Dataset")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display with Streamlit
        if dataset_version == "Start":
            
            st.markdown("""
            ### 🧠 Key Observations

            #### 🔵 Strong Positive Correlations
            - **Diagonal line**: Every variable is perfectly correlated with itself (**correlation = 1**). This is expected and appears as a deep blue diagonal.
            - **Diabetes_State and HB (Hemoglobin)**: Very strong positive correlation, suggesting high hemoglobin levels may be strongly linked to diabetic state in this dataset.
            - **HB, Cholesterol, and BMI**: All show positive correlations among themselves, which is medically logical since these are often interconnected health metrics.

            #### 🔴 Negative Correlations
            - **PhysActivity vs BMI, Heart_Disease, and GenHlth**:  
              Physical activity is slightly negatively correlated with BMI and heart disease, indicating more active individuals tend to have lower BMI and potentially better heart health.
            - **MentHlth vs DiffWalk and GenHlth**:  
              Poor mental health appears associated with walking difficulty and poor general health, which may reflect the bidirectional relationship between mental and physical health.

            #### ⚖️ Weak or No Correlations
            - **Gender** with most variables: Gender shows weak or no linear correlation with most health indicators, possibly because other factors have stronger direct effects.
            - **Diet-related features (Fruits, Veggies)**: These seem to have minimal correlation with major health indicators like Diabetes_State or Heart_Disease — linear relationships are weak, though non-linear or long-term effects might still exist.
            ### 📌 Summary
            This heatmap provides a high-level overview of how features relate to one another in a health dataset. While some strong and interpretable correlations are visible (e.g., Diabetes_State with HB), many health and lifestyle factors show only weak linear associations. This reinforces the complexity of health outcomes and the potential value of more complex modeling approaches (e.g., decision trees, neural networks) that capture non-linear interactions.

            """)

       
        else:
           
            st.markdown("""
            ### 🔍 Key Insights from the Plot

            #### ✅ Strongest Positive Correlations
            - **Diabetes_State and HB (Hemoglobin)**  
              *Interpretation:* Higher hemoglobin levels are positively associated with diabetic status. This could reflect physiological changes in individuals with diabetes, such as altered red blood cell turnover or hypoxia responses.

            - **Age and Stroke**  
              A moderate positive correlation—older age is associated with increased stroke risk, aligning with medical expectations.

            - **GenHlth and PhysHlth, MentHlth, DiffWalk**  
              These variables are interrelated as they reflect physical, mental, and general health. Poor general health is linked to worse physical health, more mental health issues, and difficulty walking.

            #### ❌ Strongest Negative Correlations
            - **Age and PhysActivity**  
              Suggests that older individuals tend to engage in less physical activity, which is consistent with aging-related limitations.

            - **Veggies and BMI, Cholesterol, Heart_Disease**  
              Weak to moderate negative correlations, implying that greater vegetable intake may be associated with healthier outcomes, though the linear relationships are not very strong.

            - **CholCheck and Stroke**  
              Slight negative correlation, which might imply that those who check their cholesterol regularly are slightly less likely to experience stroke—possibly due to better preventive care.

            #### 🧠 General Observations
            - **Weak correlations dominate the plot**: Most values are between **-0.2 and +0.2**, suggesting that many of these relationships are weak or non-linear.

            - **Lifestyle variables** like *Smoker*, *Fruits*, *Veggies*, and *HvAlcoholConsump* have minimal direct linear impact on chronic disease indicators, possibly due to:
              - Measurement limitations (e.g., self-reported intake/frequency).
              - Need for longer-term or cumulative data.
              - The influence of complex non-linear interactions, which aren’t captured by correlation alone.
            ### 📌 Summary
            This heatmap provides a foundational understanding of variable relationships in the dataset. While most correlations are weak, several important patterns emerge—especially in age-related health outcomes, diabetes predictors, and lifestyle influences on general health. However, given the weak linear correlations for many features, non-linear modeling approaches (e.g., decision trees, random forest, neural networks) may better capture the true structure of the data.
            """)

            

# ------------------ Prediction Section ------------------
def prediction_section():
    st.title("🩺 Diabetes Risk Prediction Form")
    st.markdown("Fill in the details below to record your health indicators.")
    
    model_type = st.sidebar.selectbox("Choose Model Type:", ["Classification", "Regression"])


    model_drive_ids_C = {
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
    
    model_drive_ids_R = {
        "regressor_DD": "1-tmgxZHBXoMjnyx8iHdzBORbHS43bCVj",
        "treer_DD": "1-zct0945l0EESuVkW51f60oFQXJC2AIt",
        "rfr_DD": "102J_kkpxhcdpbGOJzSXBcH5-g-RHVhXj",
        "xgbr_DD": "10A2tkOSszRryIa1PDbxyGefVj2w1-7aE",
        "mlpr_DD": "106ZKudJcHg--Q1OeFfQ-g1CF4Ps_KGpD",
        "knnr_DD": "102fam3U0c63bHIypOD6Wh_84obw6MPjg"
    }
    
    
    if model_type == "Classification":
        model_data = {
            "Model": list(model_drive_ids_C.keys()),
            "Accuracy": [0.7866, 0.7824, 0.7957, 0.7682, 0.6933, 0.7775, 0.8027, 0.8023, 0.7861],
            "Precision": [0.7589, 0.7223, 0.7641, 0.7009, 0.6064, 0.7001, 0.7470, 0.7490, 0.7589],
            "Recall": [0.6614, 0.7156, 0.6864, 0.7047, 0.6020, 0.7482, 0.7449, 0.7394, 0.6596],
            "F1 Score": [0.7068, 0.7189, 0.7232, 0.7028, 0.6042, 0.7234, 0.7460, 0.7442, 0.7058],
            "Notes": [
                "Balanced, slightly favors class 0",
                "Solid baseline; consistent class performance",
                "Strong, but no predict_proba for ROC",
                "Overfitting likely; big gap in train vs. test",
                "Very high training score (0.9825); severe overfitting",
                "Much more generalizable than single tree",
                "⚡ Best-performing model in classification",
                "Close second to XGBoost; stable performer",
                "Good baseline model, well-balanced"
            ]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Summary
        st.markdown("### 🏆 **Top Classifier Performers**")
        st.markdown("""
        - **Top Performers**: `xgb_DD` and `mlp_DD` show the best overall performance with accuracy above 0.80 and balanced precision-recall.
        - **Baseline Strong**: `logr_DD` and `pipe_DD` provide a strong logistic foundation.
        - **Overfitting Warning**: `tree_DD` massively overfits, showing a huge training score gap (0.9825 vs. 0.6933).
        - **Random Forest** (`rf_DD`) and **SVM** (`svc_DD`) are reliable but slightly behind XGB and MLP.
        - **Naive Bayes** (`gnb_DD`) is surprisingly competitive given its simplicity.
        """, unsafe_allow_html=True)
        
        model_drive_ids = model_drive_ids_C


    else:
        model_data = {
            "Model": list(model_drive_ids_R.keys()),
            "R² Score": [0.3398, -0.2955, 0.2984, 0.3928, 0.3904, 0.2679],
            "MAE": [0.3129, 0.3121, 0.3090, 0.2845, 0.2835, 0.2905],
            "RMSE": [0.3961, 0.5549, 0.4083, 0.3799, 0.3806, 0.4171],
            "Notes": [
                "Base regressor",
                "❌ Poor generalization – severe overfit",
                "Weak performance despite high training",
                "🔼 Best overall regressor",
                "Close second to XGBR",
                "Underperforms, suggests lack of complexity handling"
            ]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        model_drive_ids = model_drive_ids_R
        
        # Analysis summary
        st.markdown("### 🏆 **Top Regressor Performers**")
        st.markdown("""
        - **Best Models**: `xgbr_DD` and `mlpr_DD` outperform others in R² and MAE, providing the most accurate regression predictions.
        - **Tree-based Regression** (`treer_DD`) fails significantly on unseen data — potential severe overfitting.
        - **Random Forest Regressor** underperforms compared to expectations, possibly due to hyperparameter issues or feature importance inconsistency.
        - **Base Regression** (`regressor_DD`) serves as a useful lower bound reference.
        """, unsafe_allow_html=True)
    
    
    model_choice = st.selectbox("🔍 Select a model to use for prediction", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")
    
    def calculate_bmi(weight, height_cm):
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        return round(bmi, 2)

    def get_bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"

    # Streamlit UI
    st.markdown("#### 💪 BMI Calculator")

    # User Inputs
    weight = st.number_input("Enter your weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
    height = st.number_input("Enter your height (cm)", min_value=50.0, max_value=250.0, value=170.0)

    # Calculate button
    if st.button("Calculate BMI"):
        bmi = calculate_bmi(weight, height)
        category = get_bmi_category(bmi)

        st.success(f"Your BMI is **{bmi}**")
        st.info(f"This is considered **{category}**.")

        # Optional visual feedback
        st.progress(min(100, int(bmi * 2)))  # just for fun visual
    
    
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

def Models_Testing_section():
    
    Diabetic_DB = load_data("Cleaned")
    # Selecting features and target
    X_DD = Diabetic_DB.drop(columns=['Diabetes_State'])  # Features
    y_DD = Diabetic_DB['Diabetes_State']  # Target
        
    # Split data
    X_DD_train, X_DD_test, y_DD_train, y_DD_test = train_test_split(X_DD, y_DD, test_size = 0.2, random_state = 0)
    
    st.title("🩺 Diabetes Risk Prediction Form")
    st.markdown("Fill in the details below to record your health indicators.")
    
    model_type = st.sidebar.selectbox("Choose Model Type:", ["Classification", "Regression"])


    model_drive_ids_C = {
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
    
    model_drive_ids_R = {
        "regressor_DD": "1-tmgxZHBXoMjnyx8iHdzBORbHS43bCVj",
        "treer_DD": "1-zct0945l0EESuVkW51f60oFQXJC2AIt",
        "rfr_DD": "102J_kkpxhcdpbGOJzSXBcH5-g-RHVhXj",
        "xgbr_DD": "10A2tkOSszRryIa1PDbxyGefVj2w1-7aE",
        "mlpr_DD": "106ZKudJcHg--Q1OeFfQ-g1CF4Ps_KGpD",
        "knnr_DD": "102fam3U0c63bHIypOD6Wh_84obw6MPjg"
    }

    
    if model_type == "Classification":
        model_data = {
            "Model": list(model_drive_ids_C.keys()),
            "Accuracy": [0.7866, 0.7824, 0.7957, 0.7682, 0.6933, 0.7775, 0.8027, 0.8023, 0.7861],
            "Precision": [0.7589, 0.7223, 0.7641, 0.7009, 0.6064, 0.7001, 0.7470, 0.7490, 0.7589],
            "Recall": [0.6614, 0.7156, 0.6864, 0.7047, 0.6020, 0.7482, 0.7449, 0.7394, 0.6596],
            "F1 Score": [0.7068, 0.7189, 0.7232, 0.7028, 0.6042, 0.7234, 0.7460, 0.7442, 0.7058],
            "Notes": [
                "Balanced, slightly favors class 0",
                "Solid baseline; consistent class performance",
                "Strong, but no predict_proba for ROC",
                "Overfitting likely; big gap in train vs. test",
                "Very high training score (0.9825); severe overfitting",
                "Much more generalizable than single tree",
                "⚡ Best-performing model in classification",
                "Close second to XGBoost; stable performer",
                "Good baseline model, well-balanced"
            ]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        # Summary
        st.markdown("### 🏆 **Top Classifier Performers**")
        st.markdown("""
        - **Top Performers**: `xgb_DD` and `mlp_DD` show the best overall performance with accuracy above 0.80 and balanced precision-recall.
        - **Baseline Strong**: `logr_DD` and `pipe_DD` provide a strong logistic foundation.
        - **Overfitting Warning**: `tree_DD` massively overfits, showing a huge training score gap (0.9825 vs. 0.6933).
        - **Random Forest** (`rf_DD`) and **SVM** (`svc_DD`) are reliable but slightly behind XGB and MLP.
        - **Naive Bayes** (`gnb_DD`) is surprisingly competitive given its simplicity.
        """, unsafe_allow_html=True)
        
        model_drive_ids = model_drive_ids_C


    else:
        model_data = {
            "Model": list(model_drive_ids_R.keys()),
            "R² Score": [0.3398, -0.2955, 0.2984, 0.3928, 0.3904, 0.2679],
            "MAE": [0.3129, 0.3121, 0.3090, 0.2845, 0.2835, 0.2905],
            "RMSE": [0.3961, 0.5549, 0.4083, 0.3799, 0.3806, 0.4171],
            "Notes": [
                "Base regressor",
                "❌ Poor generalization – severe overfit",
                "Weak performance despite high training",
                "🔼 Best overall regressor",
                "Close second to XGBR",
                "Underperforms, suggests lack of complexity handling"
            ]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
        model_drive_ids = model_drive_ids_R
        
        # Analysis summary
        st.markdown("### 🏆 **Top Regressor Performers**")
        st.markdown("""
        - **Best Models**: `xgbr_DD` and `mlpr_DD` outperform others in R² and MAE, providing the most accurate regression predictions.
        - **Tree-based Regression** (`treer_DD`) fails significantly on unseen data — potential severe overfitting.
        - **Random Forest Regressor** underperforms compared to expectations, possibly due to hyperparameter issues or feature importance inconsistency.
        - **Base Regression** (`regressor_DD`) serves as a useful lower bound reference.
        """, unsafe_allow_html=True)
    
    
    model_choice = st.selectbox("🔍 Select a model to show its Testing results", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")
    

    if model is None:
        st.error("❌ Selected model could not be loaded. Please check the selection.")
        st.stop()

    print(f"🔍 Evaluating model: {model_choice}")
    train_score = model.score(X_DD_train, y_DD_train)
    test_score = model.score(X_DD_test, y_DD_test)
    y_pred = model.predict(X_DD_test)

    print(f"Training Score: {train_score:.4f}")
    print(f"Test Score    : {test_score:.4f}")
    
    def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline', line=dict(dash='dash')))
        
        fig.update_layout(
            title=f"{title} (AUC = {roc_auc:.2f})",
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(constrain='domain'),
            width=600,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Negative", "Positive"]
        
        z = cm
        x = labels
        y = labels

        z_text = [[str(y) for y in x] for x in z]

        fig = ff.create_annotated_heatmap(
            z=z, x=x, y=y, annotation_text=z_text, colorscale='Blues', showscale=True
        )
        fig.update_layout(
            title_text=title,
            xaxis=dict(title="Predicted Label"),
            yaxis=dict(title="True Label")
        )
        fig.update_layout(
            title=title,
            width=600,     # Change as needed
            height=600     # Change as needed
        )
        st.plotly_chart(fig, use_container_width=True)
    
    
    # === Classification ===
    if model_type == "Classification":
        acc = accuracy_score(y_DD_test, y_pred)
        prec = precision_score(y_DD_test, y_pred, zero_division=0)
        rec = recall_score(y_DD_test, y_pred, zero_division=0)
        f1 = f1_score(y_DD_test, y_pred, zero_division=0)
        st.markdown(f"Accuracy : {acc:.4f}")
        st.markdown(f"Precision: {prec:.4f}")
        st.markdown(f"Recall   : {rec:.4f}")
        st.markdown(f"F1 Score : {f1:.4f}")
        st.markdown("\n #### Classification Report:")
                
        st.text(classification_report(y_DD_test, y_pred, zero_division=0))


        st.markdown("### 🔍 **Confusion Matrix**")
        plot_confusion_matrix(y_DD_test, y_pred, f"{model_choice} - Confusion Matrix")

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_DD_test)[:, 1]
            st.markdown("### 🧪 **ROC Curve**")
            plot_roc_curve(y_DD_test, y_prob, f"{model_choice} - ROC Curve")
        else:
            st.warning("ROC Curve skipped: Model has no predict_proba method")

    # === Regression ===
    else:
        r2 = r2_score(y_DD_test, y_pred)
        mae = mean_absolute_error(y_DD_test, y_pred)
        mse = mean_squared_error(y_DD_test, y_pred)
        rmse = np.sqrt(mse)
        st.markdown(f"R² Score      : {r2:.4f}")
        st.markdown(f"MAE           : {mae:.4f}")
        st.markdown(f"MSE           : {mse:.4f}")
        st.markdown(f"RMSE          : {rmse:.4f}")

    print("*" * 50 + "\n")
    
    
def Prediction_Column_section():   
    
    # --- List of all input fields (same order as form inputs) ---
    fields = [
        "Diabetes_State", "HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity",
        "PhysHlth", "Gender", "Age", "Stroke", "GenHlth", "CholCheck", "Smoker",
        "Fruits", "Veggies", "HvyAlcoholConsump", "MentHlth", "DiffWalk"
    ]
    categorical_columns = [
        "Diabetes_State", "Gender", "Stroke",
        "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk",
        "HB","Cholesterol", "Heart_Disease", "PhysActivity"
    ]
    numerical_columns = [ "BMI", "MentHlth", "GenHlth", "Age", "PhysHlth"]

    # --- User selects which field to predict ---
    st.title("🧠 Health Risk Prediction App")
    target_field = st.selectbox("Select the field you want us to predict for you:", fields)
    
    if target_field == "Diabetes_State":
        
        model_drive_ids = {
            'pipe_Diabetes_State.pkl': '1j9zd7eV4GFUBrwl-hYuoXwvE7bzUZ7fo',
            'gnb_Diabetes_State.pkl': '1P_xoMjcgmMVpHdcndEJx-nygwxyIYWnK',
            'xgb_Diabetes_State.pkl': '1Ro2Z8ZikMJeLTFevHWTyJGU54Ai0ZRXn',
            'logr_Diabetes_State.pkl': '1AExNEPHjW_8KITP6qSMqLfkk6ZdkxEIF'

            }
        
    elif target_field == "HB":    
        model_drive_ids = {
            'pipe_HB.pkl': '1-w4J2D5BjPDvVqVL0ZomTCo710aJqdYv', 
            'gnb_HB.pkl': '1-vAtt1g1gNwQUxSBEVL5R2Oe2KaCkarZ', 
            'xgb_HB.pkl': '1-qx3xa1yE4GGTyYbdxxNq0NxWLLqc3Q7', 
            'logr_HB.pkl': '1-Do_j20P8iNWQc3qrvuPK3iP-8qdxSr6'
            
            }
        
        
    elif target_field == "Cholesterol":
        model_drive_ids = {
        
            'gnb_Cholesterol.pkl': '10-k31utSGKXrYTXdLdutUrn41WwE0N9Z',
            'pipe_Cholesterol.pkl': '1-yoymNnqZMOkxoriBJzV3dTIBwg_0Gmk',
            'xgb_Cholesterol.pkl': '109v8jiLqdA3xiRFh06cpv74FSDydS3Kh', 
            'logr_Cholesterol.pkl': '109mjkKw-cP8ZC65HGCJZl9dEOqoqVNue'
        
            }
        
    elif target_field == "BMI":
        model_drive_ids = {
            'xgbr_BMI.pkl': '10Cjx8Hwly0HMt4v2Swnk9t_FYvwKO6Rh', 
            'regressor_BMI.pkl': '10JiXy7bs7io6wCxC_whw9UUUshIfAOVM'
        
            }
        
    elif target_field == "Heart_Disease":
        model_drive_ids = {
            'gnb_Heart_Disease.pkl': '10Klf74O_P47GAE8jRxxvOM8lnychItSY',
            'pipe_Heart_Disease.pkl': '10RHhLSvRAbqqg4OyGSJWo-jUT8sG1rgm',
            'xgb_Heart_Disease.pkl': '10T4yjQnXR9SX-Cp7-hWEZ9nm0lgOezet', 
            'logr_Heart_Disease.pkl': '10Syu1AC9bUf3-3wYcaIELwgPTpxG8R8k'
        
            }
            
    elif target_field == "PhysActivity":
        model_drive_ids = {
            'gnb_PhysActivity.pkl': '10vODwUA4_sdkSsUQLaRmNXijLEoEL4Jw',
            'pipe_PhysActivity.pkl': '10xmQ43ZaKYxTTqKXsk9nnh_UwDmsEaoj',
            'xgb_PhysActivity.pkl': '10pi4HVlJW3Q16Ilr7lDn8FIlN6i0mJsO',
            'logr_PhysActivity.pkl': '10g0oK5azAiIR4Ujl2lbINx2Q2LVZQwBo'
        
            }
        
    elif target_field == "PhysHlth":
        model_drive_ids = {
            'regressor_PhysHlth.pkl': '1130fNlH8qW51ihHPtto_FVPp9KkkwmVf',
            'xgbr_PhysHlth.pkl': '11LF2b7EFp2lb3NVtmk-IXVPZXwG6gbL-'
        
            }
        
    elif target_field == "Gender":
        model_drive_ids = {
            'pipe_Gender.pkl': '11VGc2ndX3kXS1F_72xuwPInXCiaWeIay', 
            'gnb_Gender.pkl': '11QgBGwW_825iAxt9tXIU9qX5DUy7Ll3I', 
            'xgb_Gender.pkl': '11Q2fzJ4gtiduxGAIb0hn9NOFNDOziqh7', 
            'logr_Gender.pkl': '11MtvcVuZUSsB0Td6MPnRoNq90mZF3iNo'
        
            }
        
    elif target_field == "Age":
        model_drive_ids = {
            'regressor_Age.pkl': '11c4ETvbtzC3zJRD2RBirxheEImoAc151',
            'xgbr_Age.pkl': '11Ys0I2h7JDcpN8ZtRWu2VYJ99dy-suSC'
        
            }
        
    elif target_field == "Stroke":
        model_drive_ids = {
            'gnb_Stroke.pkl': '11jymK3AhmzTrorEUH8DaOD-xsZh2ruPm',
            'pipe_Stroke.pkl': '11jzvl26qD42hOaYTNG3iFAJp0nUPACWJ',
            'xgb_Stroke.pkl': '11ifDcMtil9iFknqtnrcppDK_Og7U4Yr5',
            'logr_Stroke.pkl': '11gZUD80wye_G6lQ5bm2SLVNR-mcwZTBC'
        
            }
        
    elif target_field == "GenHlth":
        model_drive_ids = {
            'regressor_GenHlth.pkl': '11xfX_MVcMoUWvixtyzTGSvoEX0GV8mk6',
            'xgbr_GenHlth.pkl': '11lbmxXOuwtTrX1CztPQwbu9XABsBPm5n'
        
            }
        
    elif target_field == "CholCheck":
        model_drive_ids = {
            'pipe_CholCheck.pkl': '126RtgJvf6WkRbYM_5NFn4cHsEbuh5j16',
            'gnb_CholCheck.pkl': '121UfCxyZW0sX5N0-SNAHT2u2WCyp-EyS',
            'xgb_CholCheck.pkl': '120AvQQKr_tB2RsVUtIr0RTvGCsnv9eP2',
            'logr_CholCheck.pkl': '11yRfIqVkxBhiXyCxEOGp9iO7uL1n0NS6'
            
            }
        
    elif target_field == "Smoker":
        model_drive_ids = {
            'pipe_Smoker.pkl': '12JuVvuj49IfimLfmBsz80VAmMDNhRcEs',
            'gnb_Smoker.pkl': '12GtDIFF-oN86aKlOrpX5KqHRj3HPGxID',
            'xgb_Smoker.pkl': '12Cgch_pZNQ0UeTxN_XVbeV13KaOkFowP',
            'logr_Smoker.pkl': '128UVLSCo_H4Xkc3zKSKMM0OHtJtCtW-T'
        
            }
        
    elif target_field == "Veggies":
        model_drive_ids = {
            'pipe_Veggies.pkl': '12cQFjuSSHVpmJCSk792AvdN_B-qZWzlj',
            'gnb_Veggies.pkl': '12_arDrCem_NzHTBHOuVUr4ZqWSOe67sv',
            'xgb_Veggies.pkl': '12dMoRIkpmCpB-czkdPCQAXFf6Wq2mqg3',
            'logr_Veggies.pkl': '12cbVojY0uvYAMKw49XuU388sJf8kSCCN'
        
            }
        
    elif target_field == "Fruits":
        model_drive_ids = {
            'pipe_Fruits.pkl': '12PZCsy1GbMuav0QaymmQkErvfXtfb9k6',
            'gnb_Fruits.pkl': '12LnsKNwNDTKIyKS8TGZUH02AqIZyLx1b',
            'xgb_Fruits.pkl': '12T2TtHNbWeu1sxz7qJgNfGtpz_Npvt5L',
            'logr_Fruits.pkl': '12Q510anLFqiwZ0aXDFv_35F79lvoG3Tf'
        
            }
        
    elif target_field == "HvyAlcoholConsump":
        model_drive_ids = {
            'pipe_HvyAlcoholConsump.pkl': '12deBl4Prtljy0b4nsSRCefdaukT4lAde',
            'gnb_HvyAlcoholConsump.pkl': '12eQmVKGEl8FTcDOpL47mGpWwfYbFTynC',
            'xgb_HvyAlcoholConsump.pkl': '12zzek2MfwI6i5Z8XnHUVzUSR78gh1ZYO',
            'logr_HvyAlcoholConsump.pkl': '12ympmEgrTOhmQY8GFWjrKTLayqm1FxUh'
        
            }
        
    elif target_field == "MentHlth":
        model_drive_ids = {
            'regressor_MentHlth.pkl': '12w89-QsdEtu0zlVT4ulWa5RSdMqh6Hmv',
            'xgbr_MentHlth.pkl': '13-ItGQxCAC9L4rjX-DIlVKtIwjVoOy0z'
        
            }
        
    elif target_field == "DiffWalk":
        model_drive_ids = {
            'pipe_DiffWalk.pkl': '13CLyRt_QJTFsDr9IRCnYZNfOfsvvnGQT',
            'gnb_DiffWalk.pkl': '13-njQYlzz92zrhOUERYJ5aZOcUKU2puO',
            'xgb_DiffWalk.pkl': '13D3QNzBEkCgpxGvWzIpvJ_Koogy05wY6',
            'logr_DiffWalk.pkl': '13DDuIzviI076Fj2pL0dFDTPAw9MFgkeA'
        
            }
        
     
    model_choice = st.selectbox("🔍 Select a model to show its Testing results", df_models["Model"].tolist())
    model = load_model_from_drive(model_drive_ids[model_choice])
    st.markdown(f"✅ **You selected:** `{model_choice}`")

    # --- Streamlit Form ---
    with st.form("diabetes_form"):
        st.subheader("Health Information (Fill the rest)")

        col1, col2 = st.columns(2)
        user_data = {}

        for i, field in enumerate(fields):
            if field == target_field:
                continue  # Skip input for the selected field to predict

            with (col1 if i % 2 == 0 else col2):
                if field == "Diabetes_State":
                    user_data[field] = st.radio("Diabetes State", [0, 1], format_func=lambda x: "Yes" if x else "No")
                elif field == "HB":
                    user_data[field] = st.radio("High Blood Pressure (HB)", [0, 1], format_func=lambda x: "Yes" if x else "No")
                elif field == "Cholesterol":
                    user_data[field] = st.radio("Cholesterol", [0, 1], format_func=lambda x: "High" if x else "Normal")
                elif field == "BMI":
                    user_data[field] = st.number_input("Body Mass Index (BMI)", min_value=10, max_value=70)
                elif field == "Heart_Disease":
                    user_data[field] = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
                elif field == "PhysActivity":
                    user_data[field] = st.radio("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
                elif field == "PhysHlth":
                    user_data[field] = st.slider("Poor Physical Health Days (Last 30 Days)", 0, 30)
                elif field == "Gender":
                    user_data[field] = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                elif field == "Age":
                    user_data[field] = st.number_input("Age", min_value=1, max_value=120)
                elif field == "Stroke":
                    user_data[field] = st.radio("Stroke History", [0, 1])
                elif field == "GenHlth":
                    user_data[field] = st.selectbox("General Health", [1, 2, 3, 4, 5],
                                                    format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])
                elif field == "CholCheck":
                    user_data[field] = st.radio("Cholesterol Checked", [0, 1])
                elif field == "Smoker":
                    user_data[field] = st.radio("Smoker", [0, 1])
                elif field == "Fruits":
                    user_data[field] = st.radio("Consumes Fruits", [0, 1])
                elif field == "Veggies":
                    user_data[field] = st.radio("Consumes Vegetables", [0, 1])
                elif field == "HvyAlcoholConsump":
                    user_data[field] = st.radio("Heavy Alcohol Use", [0, 1])
                elif field == "MentHlth":
                    user_data[field] = st.slider("Poor Mental Health Days", 0, 30)
                elif field == "DiffWalk":
                    user_data[field] = st.radio("Difficulty Walking", [0, 1])

        submitted = st.form_submit_button("Submit")
    
     
        
    
    # --- Prediction Logic ---
    if submitted:
        st.markdown("---")
        st.subheader("📋 Submitted Information")

        # Convert to DataFrame for model prediction
        df = pd.DataFrame([user_data])
        st.dataframe(df)
        st.success("✅ Your input has been recorded!")


        if model:
            prediction = model.predict(df)[0]
            result = "🟢 Likely Healthy" if prediction == 0 else "🔴 Likely Diabetic"
   
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
    
    
    Plotted = st.form_submit_button("Plot Model")
    if Plotted:
        # === Classification ===
        if fields in categorical_columns:
            acc = accuracy_score(y_DD_test, y_pred)
            prec = precision_score(y_DD_test, y_pred, zero_division=0)
            rec = recall_score(y_DD_test, y_pred, zero_division=0)
            f1 = f1_score(y_DD_test, y_pred, zero_division=0)
            st.markdown(f"Accuracy : {acc:.4f}")
            st.markdown(f"Precision: {prec:.4f}")
            st.markdown(f"Recall   : {rec:.4f}")
            st.markdown(f"F1 Score : {f1:.4f}")
            st.markdown("\n #### Classification Report:")
                    
            st.text(classification_report(y_DD_test, y_pred, zero_division=0))


            st.markdown("### 🔍 **Confusion Matrix**")
            plot_confusion_matrix(y_DD_test, y_pred, f"{model_choice} - Confusion Matrix")

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_DD_test)[:, 1]
                st.markdown("### 🧪 **ROC Curve**")
                plot_roc_curve(y_DD_test, y_prob, f"{model_choice} - ROC Curve")
            else:
                st.warning("ROC Curve skipped: Model has no predict_proba method")

        # === Regression ===
        else:
            r2 = r2_score(y_DD_test, y_pred)
            mae = mean_absolute_error(y_DD_test, y_pred)
            mse = mean_squared_error(y_DD_test, y_pred)
            rmse = np.sqrt(mse)
            st.markdown(f"R² Score      : {r2:.4f}")
            st.markdown(f"MAE           : {mae:.4f}")
            st.markdown(f"MSE           : {mse:.4f}")
            st.markdown(f"RMSE          : {rmse:.4f}")

        print("*" * 50 + "\n")
        

# ------------------ Main App Entry ------------------
def main():
    st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
    st.sidebar.title("🔎 Select Mode")
    mode = st.sidebar.radio("Choose a view:", ["📊 Visualizations", "🧠 Prediction","🏆 Models Testing","🧪 Choose prediction Column"])

    if mode == "📊 Visualizations":
        plotting_section()
    elif mode == "🧠 Prediction":
        prediction_section()
    elif mode == "🏆 Models Testing":
        Models_Testing_section()
    elif mode == "🧪 Choose prediction Column":
        Prediction_Column_section()
        

if __name__ == "__main__":
    main()
