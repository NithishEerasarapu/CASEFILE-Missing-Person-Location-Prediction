import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Find the main project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# File locations
input_file = BASE_DIR / "data" / "feature_engineered_gps.csv"
output_file = BASE_DIR / "data" / "clustered_gps_data.csv"
model_file = BASE_DIR / "clustering_model.pkl"
scaler_file = BASE_DIR / "clustering_scaler.pkl"

# Load data
print("Loading feature engineered GPS data...")

df = pd.read_csv(input_file)

print("Feature engineered data loaded successfully!")
print("Records:", len(df))

# Select latitude and longitude
X = df[["Latitude", "Longitude"]]

# Scale the location data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Create clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model and scaler
joblib.dump(kmeans, model_file)
joblib.dump(scaler, scaler_file)

# Save clustered dataset
df.to_csv(output_file, index=False)

print()
print("==============================")
print("K-MEANS CLUSTERING COMPLETED")
print("==============================")
print("Number of clusters: 5")
print("Model saved successfully!")
print("Clustered data saved successfully!")
print("==============================")