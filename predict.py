import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "location_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "user_encoder.pkl")
if not os.path.exists(ENCODER_PATH):
    ENCODER_PATH = os.path.join(BASE_DIR, "user_encode.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Location prediction system loaded successfully!")
print("Available User IDs:", list(encoder.classes_))

# Ask for missing person's ID
user_id = input("Enter User ID (e.g. USER_001): ").strip()

# Flexible case matching
matched_user_id = None
for class_name in encoder.classes_:
    if class_name.lower() == user_id.lower():
        matched_user_id = class_name
        break

if matched_user_id is None:
    print(f"User ID '{user_id}' not found in the dataset.")
else:
    user_encoded = encoder.transform([matched_user_id])[0]

    date = input("Enter date (YYYY-MM-DD): ").strip()
    time = input("Enter time (HH:MM): ").strip()

    date_value = pd.to_datetime(date)
    time_value = pd.to_datetime(time)

    input_data = pd.DataFrame({
        "User_ID_encoded": [user_encoded],
        "hour": [time_value.hour],
        "minute": [time_value.minute],
        "day": [date_value.day],
        "month": [date_value.month]
    })

    prediction = model.predict(input_data)
    latitude = round(float(prediction[0][0]), 6)
    longitude = round(float(prediction[0][1]), 6)

    print("\nPredicted Location")
    print("------------------")
    print("User ID:", matched_user_id)
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    print("Google Maps:", f"https://www.google.com/maps?q={latitude},{longitude}")