from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import tempfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup Flask template and static folders
template_dir = os.path.join(BASE_DIR, 'templates') if os.path.exists(os.path.join(BASE_DIR, 'templates')) else BASE_DIR
static_dir = os.path.join(BASE_DIR, 'static') if os.path.exists(os.path.join(BASE_DIR, 'static')) else BASE_DIR

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

MODEL_PATH = os.path.join(BASE_DIR, "location_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "user_encoder.pkl")
ALT_ENCODER_PATH = os.path.join(BASE_DIR, "user_encode.pkl")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_gps_data.csv")
ALT_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_gps_data.csv")

# Global in-memory model cache for serverless environments (Vercel)
model = None
encoder = None


def train_in_memory():
    """Train model in-memory (Vercel serverless read-only filesystem safe)."""
    target_csv = DATA_PATH if os.path.exists(DATA_PATH) else ALT_DATA_PATH
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Cleaned GPS dataset not found at {DATA_PATH} or {ALT_DATA_PATH}")

    df = pd.read_csv(target_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df["hour"] = pd.to_datetime(df["Time"], format="mixed").dt.hour
    df["minute"] = pd.to_datetime(df["Time"], format="mixed").dt.minute
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month

    enc = LabelEncoder()
    df["User_ID_encoded"] = enc.fit_transform(df["User_ID"])

    X = df[["User_ID_encoded", "hour", "minute", "day", "month"]]
    y = df[["Latitude", "Longitude"]]

    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X, y)

    # Safely attempt saving to disk if writable, else use temp directory
    try:
        joblib.dump(mdl, MODEL_PATH)
        joblib.dump(enc, ENCODER_PATH)
        joblib.dump(enc, ALT_ENCODER_PATH)
    except Exception:
        try:
            tmp_dir = tempfile.gettempdir()
            joblib.dump(mdl, os.path.join(tmp_dir, "location_model.pkl"))
            joblib.dump(enc, os.path.join(tmp_dir, "user_encoder.pkl"))
        except Exception:
            pass  # Keep in-memory model active

    return mdl, enc


def load_model_and_encoder():
    """Safely load or train model and encoder."""
    global model, encoder
    if model is not None and encoder is not None:
        return model, encoder

    model_valid = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0
    encoder_file = ENCODER_PATH if (os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0) else ALT_ENCODER_PATH
    encoder_valid = os.path.exists(encoder_file) and os.path.getsize(encoder_file) > 0

    if model_valid and encoder_valid:
        try:
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(encoder_file)
            return model, encoder
        except Exception as e:
            print(f"Warning: Loading existing pickle failed ({e}). Re-training in memory...")

    model, encoder = train_in_memory()
    return model, encoder


# Pre-warm model cache
try:
    model, encoder = load_model_and_encoder()
except Exception as e:
    print(f"Initialization Warning: {e}")


@app.route("/")
def home():
    try:
        mdl, enc = load_model_and_encoder()
        user_ids = list(enc.classes_)
    except Exception:
        user_ids = []
    return render_template("index.html", user_ids=user_ids)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        mdl, enc = load_model_and_encoder()

        user_id = request.form.get("user_id", "").strip()
        date = request.form.get("date", "").strip()
        time = request.form.get("time", "").strip()

        if not user_id or not date or not time:
            return render_template("index.html", error="Please fill in all fields (User ID, Date, and Time)."), 400

        # Flexible case-insensitive matching for User ID
        matched_user_id = None
        for class_name in enc.classes_:
            if class_name.lower() == user_id.lower():
                matched_user_id = class_name
                break

        if matched_user_id is None:
            available_users = ", ".join(list(enc.classes_))
            return render_template(
                "index.html",
                error=f"User ID '{user_id}' not found in dataset. Available IDs: {available_users}"
            ), 400

        user_encoded = enc.transform([matched_user_id])[0]

        date_value = pd.to_datetime(date)
        time_value = pd.to_datetime(time)

        input_data = pd.DataFrame({
            "User_ID_encoded": [user_encoded],
            "hour": [time_value.hour],
            "minute": [time_value.minute],
            "day": [date_value.day],
            "month": [date_value.month]
        })

        prediction = mdl.predict(input_data)
        latitude = round(float(prediction[0][0]), 6)
        longitude = round(float(prediction[0][1]), 6)

        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({
                "status": "success",
                "user_id": matched_user_id,
                "date": date,
                "time": time,
                "latitude": latitude,
                "longitude": longitude
            })

        try:
            return render_template(
                "result.html",
                user_id=matched_user_id,
                date=date,
                time=time,
                latitude=latitude,
                longitude=longitude
            )
        except Exception:
            return render_template(
                "index.html",
                user_id=matched_user_id,
                date=date,
                time=time,
                prediction_text=f"Predicted Location: Lat {latitude}, Lon {longitude}",
                latitude=latitude,
                longitude=longitude
            )

    except Exception as e:
        return render_template("index.html", error=f"Prediction Error: {str(e)}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
