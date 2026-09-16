import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "location_model.pkl")

features = [
    "User_ID_encoded",
    "hour",
    "minute",
    "day",
    "month"
]

importance = model.feature_importances_

result = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

result = result.sort_values(
    "Importance",
    ascending=False
)

print()
print("==============================")
print("MODEL EXPLAINABILITY")
print("==============================")
print(result.to_string(index=False))
print("==============================")

result.to_csv(
    BASE_DIR / "data" / "feature_importance.csv",
    index=False
)

print("Feature importance saved successfully!")
print("Saved as: data/feature_importance.csv")