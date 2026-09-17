import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
target_csv = os.path.join(BASE_DIR, "cleaned_gps_data.csv")
if not os.path.exists(target_csv):
    target_csv = os.path.join(BASE_DIR, "data", "cleaned_gps_data.csv")

print(f"Loading dataset from: {target_csv}")
df = pd.read_csv(target_csv)

print("Records:", len(df))

# Convert date and time
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = pd.to_datetime(df["Time"], format="mixed").dt.hour
df["minute"] = pd.to_datetime(df["Time"], format="mixed").dt.minute
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month

# Convert User_ID into numbers
encoder = LabelEncoder()
df["User_ID_encoded"] = encoder.fit_transform(df["User_ID"])

# Input features
X = df[["User_ID_encoded", "hour", "minute", "day", "month"]]

# Target: Latitude and Longitude
y = df[["Latitude", "Longitude"]]

# Create machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X, y)

# Save model and encoder
joblib.dump(model, os.path.join(BASE_DIR, "location_model.pkl"))
joblib.dump(encoder, os.path.join(BASE_DIR, "user_encoder.pkl"))
joblib.dump(encoder, os.path.join(BASE_DIR, "user_encode.pkl"))

print("Model trained successfully!")
print("Saved location_model.pkl and user_encoder.pkl")