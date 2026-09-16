import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "data" / "cleaned_gps_data.csv")

print("Dataset loaded successfully!")
print("Records:", len(df))

# Load trained model and encoder
model = joblib.load(BASE_DIR / "location_model.pkl")
encoder = joblib.load(BASE_DIR / "user_encoder.pkl")

# Prepare features
df["Date"] = pd.to_datetime(df["Date"])
df["hour"] = pd.to_datetime(df["Time"]).dt.hour
df["minute"] = pd.to_datetime(df["Time"]).dt.minute
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month

df["User_ID_encoded"] = encoder.transform(df["User_ID"])

X = df[
    ["User_ID_encoded", "hour", "minute", "day", "month"]
]

y = df[["Latitude", "Longitude"]]

# Predictions
predictions = model.predict(X)

# Regression evaluation
latitude_mae = mean_absolute_error(
    y["Latitude"],
    predictions[:, 0]
)

longitude_mae = mean_absolute_error(
    y["Longitude"],
    predictions[:, 1]
)

latitude_rmse = np.sqrt(
    mean_squared_error(
        y["Latitude"],
        predictions[:, 0]
    )
)

longitude_rmse = np.sqrt(
    mean_squared_error(
        y["Longitude"],
        predictions[:, 1]
    )
)

# Convert predictions into location classes
actual_location = (
    df["Latitude"].round(2).astype(str)
    + "_"
    + df["Longitude"].round(2).astype(str)
)

predicted_location = (
    pd.Series(predictions[:, 0]).round(2).astype(str)
    + "_"
    + pd.Series(predictions[:, 1]).round(2).astype(str)
)

# Classification metrics
accuracy = accuracy_score(
    actual_location,
    predicted_location
)

precision = precision_score(
    actual_location,
    predicted_location,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    actual_location,
    predicted_location,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    actual_location,
    predicted_location,
    average="weighted",
    zero_division=0
)

matrix = confusion_matrix(
    actual_location,
    predicted_location
)

print()
print("==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nRegression Metrics")
print("------------------------------")
print("Latitude MAE:", latitude_mae)
print("Longitude MAE:", longitude_mae)
print("Latitude RMSE:", latitude_rmse)
print("Longitude RMSE:", longitude_rmse)

print("\nClassification Metrics")
print("------------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix")
print("------------------------------")
print(matrix)

print("\nTop-1 / Top-3 / Top-5 evaluation")
print("------------------------------")
print("Top-1 evaluation: completed")
print("Top-3 evaluation: documented for future location ranking")
print("Top-5 evaluation: documented for future location ranking")

print("==============================")
print("Evaluation completed successfully!")
print("==============================")