import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("📊 E-commerce Customer Segmentation System")
st.markdown("Identify your **Loyal**, **Potential**, and **At-Risk** customers using Machine Learning.")

# 2. Load Models & Data
@st.cache_data
def load_data():
    df = pd.read_csv('rfm_analysis.csv')
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return df, kmeans, scaler

try:
    df, kmeans, scaler = load_data()
except FileNotFoundError:
    st.error("Error: Model files not found! Please make sure .pkl and .csv files are in the same folder.")
    st.stop()

# Cluster Mapping (তোমার নোটবুকের রেজাল্ট অনুযায়ী)
# 0: Potential, 1: Risk, 2: Loyal
cluster_names = {
    0: 'Potential Customer',
    1: 'At Risk Customer',
    2: 'Loyal/Best Customer'
}

# 3. Sidebar: Predict New Customer
st.sidebar.header("🔍 Check New Customer")
st.sidebar.write("Enter customer details below:")

recency = st.sidebar.number_input("Days Since Last Visit (Recency)", min_value=1, value=10)
frequency = st.sidebar.number_input("Total Transactions (Frequency)", min_value=1, value=5)
monetary = st.sidebar.number_input("Total Spent ($) (Monetary)", min_value=1.0, value=500.0)

if st.sidebar.button("Predict Segment"):
    # Preprocessing (Log Transform -> Scale -> Predict)
    # Note: We need to pass data in the same format as training
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    # Log transform
    input_log = np.log1p(input_data)
    
    # Scaling
    input_scaled = scaler.transform(input_log)
    
    # Prediction
    prediction = kmeans.predict(input_scaled)[0]
    segment_name = cluster_names[prediction]
    
    st.sidebar.success(f"This customer belongs to: **{segment_name}**")

# 4. Main Dashboard
st.write("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution")
    # Mapping cluster numbers to names for the plot
    df['Segment_Name'] = df['Cluster'].map(cluster_names)
    
    fig1, ax1 = plt.figure(figsize=(8, 5)), plt.gca()
    sns.countplot(x='Segment_Name', data=df, palette='viridis', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col2:
    st.subheader("Segment Analysis (Recency vs Spending)")
    fig2, ax2 = plt.figure(figsize=(8, 5)), plt.gca()
    sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Segment_Name', palette='viridis', ax=ax2)
    plt.title("Recency vs Monetary")
    st.pyplot(fig2)

# 5. Show Data Table
st.write("---")
st.subheader("Customer Data View")
st.dataframe(df.head(10))