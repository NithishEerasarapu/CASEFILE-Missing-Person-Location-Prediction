from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("location_model.pkl")
encoder = joblib.load("user_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    user_id = request.form["user_id"]
    date = request.form["date"]
    time = request.form["time"]

    if user_id not in encoder.classes_:
        return "User ID not found in dataset."

    user_encoded = encoder.transform([user_id])[0]

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

    latitude = prediction[0][0]
    longitude = prediction[0][1]

    return render_template(
        "result.html",
        user_id=user_id,
        latitude=latitude,
        longitude=longitude
    )


if __name__ == "__main__":
    app.run(debug=True)