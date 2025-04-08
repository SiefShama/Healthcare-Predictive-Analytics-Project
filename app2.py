import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
import os

# Load the data using relative path
@st.cache_data
def load_data():
    file_path = os.path.join(os.getcwd(), 'data', 'Diabetic_DB_Mil2.csv')
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found at {file_path}")
        return pd.DataFrame()

# Load the data
Diabetic_DB = load_data()

# Categorical and Numerical Columns
categorical_columns = [
    "Diabetes_State", "PhysHlth", "Gender", "Age", "Stroke", "GenHlth",
    "CholCheck", "Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "DiffWalk"
]

numerical_columns = ["HB", "Cholesterol", "BMI", "Heart_Disease", "PhysActivity", "MentHlth"]

# Plot for a single categorical column
def plot_categorical(col):
    count_data = Diabetic_DB[col].value_counts().reset_index()
    count_data.columns = [col, "Count"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=count_data[col],
            y=count_data["Count"],
            marker=dict(color=count_data["Count"], colorscale="RdBu"),
        )
    ])
    
    fig.update_layout(
        title=f"Distribution of {col}",
        xaxis_title=col,
        yaxis_title="Count"
    )
    
    return fig

# Plot for a single numerical column
def plot_numerical(col):
    data = Diabetic_DB[col].dropna()
    
    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            name="Histogram",
            marker_color='royalblue',
            opacity=0.7,
            histnorm='probability density'
        )
    )

    # KDE
    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 200)
    y_vals = kde(x_vals)

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='KDE',
            line=dict(color='crimson')
        )
    )

    fig.update_layout(
        title=f"Histogram with KDE for {col}",
        xaxis_title=col,
        yaxis_title="Density",
        barmode='overlay'
    )

    return fig

# Streamlit App Layout
def main():
    st.title("Diabetes Dataset Visualizations")
    st.sidebar.header("Visualization Options")

    plot_type = st.sidebar.selectbox("Choose plot type:", ["Categorical", "Numerical"])

    if plot_type == "Categorical":
        selected_col = st.selectbox("Select a categorical column:", categorical_columns)
        st.plotly_chart(plot_categorical(selected_col))

    elif plot_type == "Numerical":
        selected_col = st.selectbox("Select a numerical column:", numerical_columns)
        st.plotly_chart(plot_numerical(selected_col))

# Run the app
if __name__ == "__main__":
    main()
