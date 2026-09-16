import pandas as pd
import numpy as np

# Load cleaned GPS data
df = pd.read_csv("data/cleaned_gps_data.csv")

print("Cleaned GPS data loaded!")
print("Records:", len(df))

# Convert date and time
df["Date"] = pd.to_datetime(df["Date"])
df["Time"] = pd.to_datetime(df["Time"])

# Sort movement records
df = df.sort_values(["User_ID", "Date", "Time"])

# Time features
df["hour"] = df["Time"].dt.hour
df["minute"] = df["Time"].dt.minute
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["weekday"] = df["Date"].dt.dayofweek
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# Previous GPS coordinates
df["previous_latitude"] = df.groupby("User_ID")["Latitude"].shift(1)
df["previous_longitude"] = df.groupby("User_ID")["Longitude"].shift(1)

# Distance between consecutive GPS points
lat_difference = df["Latitude"] - df["previous_latitude"]
lon_difference = df["Longitude"] - df["previous_longitude"]

df["distance"] = np.sqrt(
    lat_difference ** 2 + lon_difference ** 2
)

# Speed approximation
df["speed"] = df["distance"]

# Replace missing values created by first record
df["distance"] = df["distance"].fillna(0)
df["speed"] = df["speed"].fillna(0)

# Save engineered dataset
df.to_csv("data/feature_engineered_gps.cvs")