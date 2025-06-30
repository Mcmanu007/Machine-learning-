import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="🌾 Crop Recommendation App", layout="wide")
st.title("🌿 Smart Crop Recommendation System")
st.markdown("""
Welcome to the Crop Recommendation System. Based on soil and environmental conditions like **Nitrogen**, **Phosphorus**, **Potassium**, **Temperature**, **Humidity**, **pH**, and **Rainfall**, this system suggests the most suitable crop to cultivate.
""")

# Load pre-trained model
@st.cache_resource
def load_model():
    with open("crop_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Prediction"])

if option == "Prediction":
    st.subheader("🤖 Predict the Suitable Crop")
    
    # Input features
    N = st.number_input("Nitrogen", 0, 140, 90)
    P = st.number_input("Phosphorus", 5, 145, 42)
    K = st.number_input("Potassium", 5, 205, 43)
    temperature = st.number_input("Temperature (°C)", 8.0, 45.0, 25.0)
    humidity = st.number_input("Humidity (%)", 10.0, 100.0, 80.0)
    ph = st.number_input("pH", 3.5, 9.5, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, 100.0)

    if st.button("Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        st.success(f"🌱 Recommended Crop: **{prediction[0].capitalize()}**")
