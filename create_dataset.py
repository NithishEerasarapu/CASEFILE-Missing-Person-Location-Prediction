import pandas as pd
import numpy as np

# Create random GPS data
np.random.seed(42)

data = []

for i in range(500):

    latitude = 19.07 + np.random.uniform(-0.05, 0.05)
    longitude = 72.87 + np.random.uniform(-0.05, 0.05)

    data.append({
        "User_ID": "USER_001",
        "Latitude": latitude,
        "Longitude": longitude,
        "Date": "2026-09-15",
        "Time": f"{np.random.randint(6, 23)}:{np.random.randint(0, 60):02d}:00"
    })

# Convert data into a table
df = pd.DataFrame(data)

# Save the dataset inside the data folder
df.to_csv("data/gps_data.csv", index=False)

print("GPS dataset created successfully!")
print("Number of records:", len(df))
print(df.head())