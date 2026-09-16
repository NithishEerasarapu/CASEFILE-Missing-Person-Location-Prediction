import pandas as pd
import joblib
from pathlib import Path

# Find the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load clustered GPS data
input_file = BASE_DIR / "data" / "clustered_gps_data.csv"
df = pd.read_csv(input_file)

print("GPS data loaded successfully!")
print("Records:", len(df))

# Create a location/state from each cluster
df["Location"] = df["Cluster"].astype(str)

# Sort movement records
df["Date"] = pd.to_datetime(df["Date"])
df["Time"] = pd.to_datetime(df["Time"])

df = df.sort_values(["User_ID", "Date", "Time"])

# Create previous location
df["Previous_Location"] = df.groupby("User_ID")["Location"].shift(1)

# Remove first record of each user
transitions = df.dropna(
    subset=["Previous_Location"]
)

# Create transition table
transition_table = pd.crosstab(
    transitions["Previous_Location"],
    transitions["Location"],
    normalize="index"
)

print()
print("==============================")
print("MARKOV CHAIN ROUTE ANALYSIS")
print("==============================")
print("Transition probabilities:")
print(transition_table)

# Save transition probabilities
transition_table.to_csv(
    BASE_DIR / "data" / "route_transition_probabilities.csv"
)

print()
print("Markov Chain route prediction completed!")
print("Transition table saved successfully!")
print("==============================")