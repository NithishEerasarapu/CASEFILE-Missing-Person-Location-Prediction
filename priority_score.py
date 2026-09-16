import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "anomaly_detected_gps.csv"
output_file = BASE_DIR / "data" / "priority_scored_gps.csv"

df = pd.read_csv(input_file)

print("Anomaly data loaded successfully!")
print("Records:", len(df))

# ML-based probability score
df["ML_Probability"] = 70

# Historical frequency
df["Historical_Frequency"] = 60

# Route similarity
df["Route_Similarity"] = 50

# Distance relevance
df["Distance_Relevance"] = 60

# Time relevance
df["Time_Relevance"] = 70

# Anomaly evidence
df["Anomaly_Evidence"] = df["Anomaly"].apply(
    lambda x: 30 if x == -1 else 10
)

# Weighted priority score
df["Priority_Score"] = (
    df["ML_Probability"] * 0.30
    + df["Historical_Frequency"] * 0.20
    + df["Route_Similarity"] * 0.15
    + df["Distance_Relevance"] * 0.15
    + df["Time_Relevance"] * 0.10
    + df["Anomaly_Evidence"] * 0.10
)

# Priority category
def priority_category(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Very High"

df["Priority_Level"] = df["Priority_Score"].apply(
    priority_category
)

df.to_csv(output_file, index=False)

print()
print("==============================")
print("PRIORITY SCORE COMPLETED")
print("==============================")
print("Priority score calculated successfully!")
print("Results saved as:", output_file)
print("==============================")