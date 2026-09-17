from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoder
MODEL_PATH = os.path.join(BASE_DIR, "location_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "user_encoder.pkl")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
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
            "day": [date_value.day],
            "month": [date_value.month],
            "weekday": [date_value.weekday()]
        })

        prediction = model.predict(input_data)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Location: {prediction[0]}"
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
