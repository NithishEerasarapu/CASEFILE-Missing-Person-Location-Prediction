import pandas as pd
import folium
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "priority_scored_gps.csv"
output_file = BASE_DIR / "data" / "casefile_map.html"

df = pd.read_csv(input_file)

print("Priority data loaded successfully!")
print("Records:", len(df))

# Center map around the average GPS location
center_lat = df["Latitude"].mean()
center_lon = df["Longitude"].mean()

case_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12
)

# Add GPS points
for _, row in df.head(200).iterrows():

    label = "Normal"

    if row["Anomaly"] == -1:
        label = "Anomaly"

    popup_text = (
        f"User ID: {row['User_ID']}<br>"
        f"Latitude: {row['Latitude']:.6f}<br>"
        f"Longitude: {row['Longitude']:.6f}<br>"
        f"Cluster: {row['Cluster']}<br>"
        f"Status: {label}<br>"
        f"Priority: {row['Priority_Level']}"
    )

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        popup=folium.Popup(popup_text, max_width=300),
        fill=True
    ).add_to(case_map)

case_map.save(output_file)

print()
print("==============================")
print("INTERACTIVE MAP CREATED")
print("==============================")
print("Map saved as:", output_file)
print("==============================")