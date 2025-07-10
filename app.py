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
option = st.sidebar.radio("Go to", ["Home",
                                    "Prediction",
                                    "Cost-Benefit Calculator", 
                                    "Farm Diary"
                                    ], index=0)

if option == "Home":
    st.subheader("🏡 Home")
    st.write("""
    🌾 Smart Crop Recommender

Welcome to the **Smart Crop Recommender App**!  
This application uses machine learning to recommend the most suitable crops to plant based on various environmental factors. By analyzing soil and climate conditions, our model helps farmers make data-driven decisions to improve agricultural productivity and sustainability.

---

## 🤖 How It Works

The model has been trained on extensive historical agricultural data, capturing patterns between crop performance and key environmental variables such as:

- Soil type and pH
- Rainfall levels
- Temperature ranges
- Humidity

Based on your inputs, the system instantly suggests the best crops to grow in your specific region and conditions.

---

## 🚀 Features

- **Input environmental data:** Easily enter details like soil type, rainfall, temperature, and more.
- **Accurate crop suggestions:** Get instant recommendations backed by data and machine learning algorithms.
- **User-friendly interface:** Designed for simplicity and ease of use, even for first-time users.
- **Educational insights:** Understand why certain crops are recommended and learn best practices for planting.

---

## 🌍 Why This Matters — Especially in Africa

Agriculture is the backbone of many African economies, employing over 60% of the population. Yet, many farmers still rely on intuition or outdated knowledge, leading to:

- Low yields
- Crop failures
- Food insecurity
- Economic losses

This app empowers **smallholder farmers** and agricultural planners by:

- 🧠 Making smart decisions based on **science, not guesswork**
- 📈 Increasing **crop yield** and **profitability**
- 🌱 Promoting **sustainable farming** by matching crops to local environments
- 💡 Reducing **trial-and-error planting** and associated costs
- 🛡️ Strengthening food security across the continent

---

## 🤝 A Tool for the Future

By equipping farmers with accurate crop predictions, we pave the way for:

- Climate-resilient agriculture
- Reduced hunger and poverty
- Smarter land use
- Increased digital adoption in rural areas

Whether you're a farmer, an agricultural extension officer, or a policymaker, this tool provides **actionable insights** to improve outcomes for individuals and communities.

---

Ready to grow smarter? 🌾  
Head over to the **Prediction** page to get started!""")
    
if option == "Cost-Benefit Calculator":
    st.subheader("💰 Crop Cost-Benefit Calculator")

    crop = st.selectbox("Select Crop", [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton",
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans",
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
])


    st.markdown("### 💸 Enter Estimated Costs (GHS)")
    seed = st.number_input("Seed Cost")
    fert = st.number_input("Fertilizer Cost")
    labor = st.number_input("Labor Cost")
    other = st.number_input("Other Costs")

    st.markdown("### 📈 Yield & Market Info")
    yield_kg = st.number_input("Expected Yield (kg)")
    price_per_kg = st.number_input("Price per kg")

    if st.button("Calculate Profit"):
        total_cost = seed + fert + labor + other
        revenue = yield_kg * price_per_kg
        profit = revenue - total_cost
        margin = (profit / revenue) * 100 if revenue else 0

        st.success(f"Net Profit: GHS {profit:.2f}")
        st.info(f"Profit Margin: {margin:.2f}%")
        st.write("### Breakdown of Costs")
        st.write(f"**Total Costs:** GHS {total_cost:.2f}")
        st.write(f"**Total Revenue:** GHS {revenue:.2f}")
        
        
if option == "Farm Diary":
    import datetime
    import pandas as pd

    st.subheader("📒 Digital Farm Diary")

    if "diary" not in st.session_state:
        st.session_state.diary = []

    date = st.date_input("Date", datetime.date.today())
    activity = st.selectbox("Activity", ["Planting", "Fertilizer", "Irrigation", "Harvest", "Other"])
    notes = st.text_area("Notes or Description")

    if st.button("Add Entry"):
        st.session_state.diary.append({
            "Date": date,
            "Activity": activity,
            "Notes": notes
        })
        st.success("Added to your diary.")

    if st.session_state.diary:
        st.subheader("📅 Logged Activities")
        df = pd.DataFrame(st.session_state.diary)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Diary", csv, "farm_diary.csv", "text/csv")


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

        df_input = pd.DataFrame([{
            "Nitrogen": N,
            "Phosphorus": P,
            "Potassium": K,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "pH": ph,
            "Rainfall (mm)": rainfall,
            "Predicted Crop": prediction[0].capitalize()
        }])

        csv = df_input.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Input Data as CSV",
            data=csv,
            file_name="crop_input_data.csv",
            mime="text/csv"
        )
