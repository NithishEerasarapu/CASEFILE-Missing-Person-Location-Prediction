from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates_path = os.path.join(BASE_DIR, 'templates')
static_path = os.path.join(BASE_DIR, 'static')

app = Flask(
    __name__,
    template_folder=templates_path if os.path.exists(templates_path) else BASE_DIR,
    static_folder=static_path if os.path.exists(static_path) else BASE_DIR
)

# Global model & encoder cache for serverless environments
MODEL = None
ENCODER = None


def get_dataset_path():
    p1 = os.path.join(BASE_DIR, "cleaned_gps_data.csv")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DIR, "data", "cleaned_gps_data.csv")
    if os.path.exists(p2):
        return p2
    return "cleaned_gps_data.csv"


def train_model_in_memory():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    csv_path = get_dataset_path()
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["hour"] = pd.to_datetime(df["Time"], format="mixed").dt.hour
    df["minute"] = pd.to_datetime(df["Time"], format="mixed").dt.minute
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month

    enc = LabelEncoder()
    df["User_ID_encoded"] = enc.fit_transform(df["User_ID"])

    X = df[["User_ID_encoded", "hour", "minute", "day", "month"]]
    y = df[["Latitude", "Longitude"]]

    mdl = RandomForestRegressor(n_estimators=50, random_state=42)
    mdl.fit(X, y)
    return mdl, enc


def get_model_and_encoder():
    global MODEL, ENCODER
    if MODEL is not None and ENCODER is not None:
        return MODEL, ENCODER

    # Try loading pickled model/encoder
    model_path = os.path.join(BASE_DIR, "location_model.pkl")
    encoder_path = os.path.join(BASE_DIR, "user_encoder.pkl")
    if not os.path.exists(encoder_path):
        encoder_path = os.path.join(BASE_DIR, "user_encode.pkl")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 100 and os.path.exists(encoder_path) and os.path.getsize(encoder_path) > 100:
        try:
            MODEL = joblib.load(model_path)
            ENCODER = joblib.load(encoder_path)
            return MODEL, ENCODER
        except Exception as e:
            print(f"Pickle load error: {e}. Fallback to in-memory training...")

    MODEL, ENCODER = train_model_in_memory()
    return MODEL, ENCODER


@app.route("/")
def home():
    try:
        _, enc = get_model_and_encoder()
        user_ids = list(enc.classes_)
    except Exception:
        user_ids = ["USER_001"]

    try:
        return render_template("index.html", user_ids=user_ids)
    except Exception:
        return """
        <!DOCTYPE html>
        <html>
        <head><title>CASEFILE - Location Prediction</title></head>
        <body style="font-family:sans-serif; background:#07111f; color:#fff; padding:40px; text-align:center;">
            <h1>🕵️ CASEFILE - Location Prediction</h1>
            <form action="/predict" method="POST" style="margin-top:30px;">
                <label>User ID:</label><br>
                <input type="text" name="user_id" value="USER_001" required style="padding:10px; margin:10px; width:250px;"><br>
                <label>Date:</label><br>
                <input type="date" name="date" required style="padding:10px; margin:10px; width:250px;"><br>
                <label>Time:</label><br>
                <input type="time" name="time" required style="padding:10px; margin:10px; width:250px;"><br>
                <button type="submit" style="padding:12px 24px; background:#2563eb; color:#fff; border:none; border-radius:6px; cursor:pointer;">Predict Location</button>
            </form>
        </body>
        </html>
        """


@app.route("/predict", methods=["POST"])
def predict():
    try:
        mdl, enc = get_model_and_encoder()

        user_id = request.form.get("user_id", "").strip()
        date = request.form.get("date", "").strip()
        time = request.form.get("time", "").strip()

        if not user_id or not date or not time:
            return "Error: Missing required fields (User ID, Date, or Time).", 400

        matched_user_id = None
        for class_name in enc.classes_:
            if class_name.lower() == user_id.lower():
                matched_user_id = class_name
                break

        if matched_user_id is None:
            available = ", ".join(list(enc.classes_))
            return f"Error: User ID '{user_id}' not found in dataset. Available IDs: {available}", 400

        user_encoded = enc.transform([matched_user_id])[0]
        date_val = pd.to_datetime(date)
        time_val = pd.to_datetime(time)

        input_df = pd.DataFrame({
            "User_ID_encoded": [user_encoded],
            "hour": [time_val.hour],
            "minute": [time_val.minute],
            "day": [date_val.day],
            "month": [date_val.month]
        })

        pred = mdl.predict(input_df)
        lat = round(float(pred[0][0]), 6)
        lon = round(float(pred[0][1]), 6)

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({
                "status": "success",
                "user_id": matched_user_id,
                "date": date,
                "time": time,
                "latitude": lat,
                "longitude": lon
            })

        try:
            return render_template(
                "result.html",
                user_id=matched_user_id,
                date=date,
                time=time,
                latitude=lat,
                longitude=lon
            )
        except Exception:
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Prediction Result</title></head>
            <body style="font-family:sans-serif; background:#07111f; color:#fff; padding:40px; text-align:center;">
                <h1>📍 Prediction Result</h1>
                <p><strong>User ID:</strong> {matched_user_id}</p>
                <p><strong>Date & Time:</strong> {date} {time}</p>
                <p><strong>Predicted Latitude:</strong> {lat}</p>
                <p><strong>Predicted Longitude:</strong> {lon}</p>
                <p><a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" style="color:#38bdf8;">🗺️ Open in Google Maps</a></p>
                <a href="/" style="color:#aaa;">← Back</a>
            </body>
            </html>
            """
    except Exception as e:
        err_msg = traceback.format_exc()
        return f"<h3>Internal Server Error</h3><pre>{err_msg}</pre>", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
