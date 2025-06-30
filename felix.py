# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
data = pd.read_csv("personality_dataset.csv")
data["Stage_fear"] = data["Stage_fear"].map({"Yes": 1, "No": 0})
data["Drained_after_socializing"] = data["Drained_after_socializing"].map({"Yes": 1, "No": 0})

mean_cols = [
    "Time_spent_Alone", "Stage_fear", "Social_event_attendance",
    "Going_outside", "Drained_after_socializing", "Friends_circle_size", "Post_frequency"
]
for col in mean_cols:
    data[col].fillna(data[col].mean(), inplace=True)

X = data[mean_cols]
y = data["Personality"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("personality_model.pkl", "wb") as f:
    pickle.dump(model, f)
