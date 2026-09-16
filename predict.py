import pandas as pd
import joblib

# Load trained model
model = joblib.load("location_model.pkl")
encoder = joblib.load("user_encoder.pkl")

print("Location prediction system loaded successfully!")

# Ask for missing person's ID
user_id = input("Enter User ID: ")

# Check whether User ID exists
if user_id not in encoder.classes_:
    print("User ID not found in the dataset.")
else:
    # Encode User ID
    user_encoded = encoder.transform([user_id])[0]

    # Ask for date and time
    date = input("Enter date (YYYY-MM-DD): ")
    time = input("Enter time (HH:MM): ")

    date_value = pd.to_datetime(date)
    time_value = pd.to_datetime(time)

    # Create input data
    input_data = pd.DataFrame({
        "User_ID_encoded": [user_encoded],
        "hour": [time_value.hour],
        "minute": [time_value.minute],
        "day": [date_value.day],
        "month": [date_value.month]
    })

    # Predict location
    prediction = model.predict(input_data)

    latitude = prediction[0][0]
    longitude = prediction[0][1]

    print("\nPredicted Location")
    print("------------------")
    print("Latitude:", latitude)
    print("Longitude:", longitude)