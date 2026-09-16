import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Find the clustered GPS file
input_file = BASE_DIR / "data" / "clustered_gps_data.csv"

print("Looking for:", input_file)

if not input_file.exists():
    print("ERROR: clustered_gps_data.csv was not found!")
    print("Please check the data folder.")
    exit()

df = pd.read_csv(input_file)

print("Clustered GPS data loaded successfully!")
print("Records:", len(df))

# Features for anomaly detection
X = df[["Latitude", "Longitude", "distance", "speed"]]

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X)

df["Anomaly_Label"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Save model
joblib.dump(model, BASE_DIR / "anomaly_model.pkl")

# Save results
output_file = BASE_DIR / "data" / "anomaly_detected_gps.csv"
df.to_csv(output_file, index=False)

print("==============================")
print("ANOMALY DETECTION COMPLETED")
print("==============================")
print("Normal records:", (df["Anomaly"] == 1).sum())
print("Anomaly records:", (df["Anomaly"] == -1).sum())
print("Results saved:", output_file)