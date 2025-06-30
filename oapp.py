import streamlit as st
import pandas as pd
import pickle

import os

def main():
    st.set_page_config(page_title="Personality Predictor", layout="centered")
    st.title("🧠 Personality Predictor App")
    st.markdown("Provide your behavioral inputs to predict your personality type!")

    # Load pre-trained model

model_path = os.path.join(os.path.dirname(__file__), 'personality_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)




    st.sidebar.header("📋 Input Your Details")

    # Sidebar inputs
    user_input = pd.DataFrame([{
        "Time_spent_Alone": st.sidebar.slider("⏰ Time Spent Alone (hours/week)", 0, 100, 10),
        "Stage_fear": 1 if st.sidebar.selectbox("😰 Do you have stage fear?", ["Yes", "No"]) == "Yes" else 0,
        "Social_event_attendance": st.sidebar.slider("🎉 Events Attended/Month", 0, 30, 5),
        "Going_outside": st.sidebar.slider("🌳 Going Outside/Week", 0, 14, 7),
        "Drained_after_socializing": 1 if st.sidebar.selectbox("😓 Feel drained after socializing?", ["Yes", "No"]) == "Yes" else 0,
        "Friends_circle_size": st.sidebar.slider("👥 Friend Circle Size", 0, 50, 10),
        "Post_frequency": st.sidebar.slider("📱 Posts/Week", 0, 50, 5)
    }])

    # Only define and use prediction inside the button logic
    if st.sidebar.button("🔮 Predict Personality"):
        prediction = model.predict(user_input)[0]
        st.success(f"### 🎯 Predicted Personality: **{prediction}**")
        st.balloons()

        # Optional message
        if prediction.lower().startswith("intro"):
            st.info("You seem reflective and enjoy solitude. 📚")
        elif prediction.lower().startswith("extro"):
            st.info("You're outgoing and thrive in social settings! 🎉")
        else:
            st.info("You have a unique mix of traits! 🌈")

# Run
if __name__ == "__main__":
    main()
