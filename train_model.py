import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
df = pd.read_csv("data/cleaned_gps_data.csv")

print("Cleaned dataset loaded!")
print("Records:", len(df))

# Convert date and time
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = pd.to_datetime(df["Time"]).dt.hour
df["minute"] = pd.to_datetime(df["Time"]).dt.minute
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month

# Convert User_ID into numbers
encoder = LabelEncoder()
df["User_ID_encoded"] = encoder.fit_transform(df["User_ID"])

# Input features
X = df[
    [
        "User_ID_encoded",
        "hour",
        "minute",
        "day",
        "month"
    ]
]

# Target: Latitude and Longitude
y = df[["Latitude", "Longitude"]]

# Create machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

# Save model
joblib.dump(model, "location_model.pkl")

# Save encoder
joblib.dump(encoder, "user_encoder.pkl")

print("Model trained successfully!")
print("Model saved as: location_model.pkl")
print("Encoder saved as: user_encoder.pkl")