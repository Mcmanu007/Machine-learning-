import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv("Crop_recommendation.csv")
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to pickle
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to crop_model.pkl")
