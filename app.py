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
        <head><title>CASEFILE</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet"></head>
        <body style="font-family:'Inter',sans-serif; background:#0a0a0a; color:#e5e5e5; margin:0; padding:50px 20px;">
            <div style="max-width:460px; margin:auto;">
                <h1 style="text-align:center; font-size:28px; color:#fff; margin-bottom:8px;">🔍 CASEFILE</h1>
                <p style="text-align:center; color:#525252; font-size:14px; margin-bottom:30px;">Missing Person Location Prediction</p>
                <div style="background:#141414; border:1px solid #262626; border-radius:16px; padding:32px;">
                    <form action="/predict" method="POST">
                        <label style="display:block; font-size:12px; color:#a3a3a3; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Person ID</label>
                        <input type="text" name="user_id" value="USER_001" required style="width:100%; padding:12px; background:#0a0a0a; color:#fff; border:1px solid #262626; border-radius:8px; font-size:14px; margin-bottom:16px; outline:none;">
                        <label style="display:block; font-size:12px; color:#a3a3a3; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Date</label>
                        <input type="date" name="date" required style="width:100%; padding:12px; background:#0a0a0a; color:#fff; border:1px solid #262626; border-radius:8px; font-size:14px; margin-bottom:16px; outline:none;">
                        <label style="display:block; font-size:12px; color:#a3a3a3; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">Time</label>
                        <input type="time" name="time" required style="width:100%; padding:12px; background:#0a0a0a; color:#fff; border:1px solid #262626; border-radius:8px; font-size:14px; margin-bottom:20px; outline:none;">
                        <button type="submit" style="width:100%; padding:14px; background:#2563eb; color:#fff; border:none; border-radius:8px; font-size:14px; font-weight:700; cursor:pointer;">Predict Location →</button>
                    </form>
                </div>
            </div>
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
            <html><head><title>Result - CASEFILE</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
            <style>
                * {{ margin:0; padding:0; box-sizing:border-box; }}
                body {{ font-family:'Inter',sans-serif; background:#0a0a0a; color:#e5e5e5; margin:0; padding:50px 20px; }}
                .w {{ max-width:520px; margin:auto; }}
                .hd {{ text-align:center; margin-bottom:28px; }}
                .si {{ width:50px; height:50px; background:linear-gradient(135deg,#065f46,#064e3b); border:2px solid #10b981; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:22px; margin:0 auto 14px; }}
                .hd h1 {{ font-size:26px; font-weight:800; color:#fff; margin-bottom:4px; }}
                .hd p {{ font-size:13px; color:#525252; }}
                .rc {{ background:#141414; border:1px solid #262626; border-radius:16px; overflow:hidden; margin-bottom:18px; }}
                .rs {{ padding:22px 24px; border-top:1px solid #1a1a1a; }}
                .rs:first-child {{ border-top:none; }}
                .sl {{ font-size:11px; font-weight:700; color:#525252; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:14px; }}
                .dr {{ display:flex; justify-content:space-between; align-items:center; padding:10px 0; }}
                .dr+.dr {{ border-top:1px solid #1a1a1a; }}
                .dk {{ font-size:14px; color:#737373; font-weight:500; }}
                .dv {{ font-size:14px; color:#fff; font-weight:600; }}
                .cg {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
                .cb {{ background:#0a0a0a; border:1px solid #262626; border-radius:12px; padding:18px; text-align:center; }}
                .cl {{ font-size:11px; color:#525252; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }}
                .cv {{ font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:700; color:#3b82f6; }}
                .cu {{ font-size:13px; color:#525252; }}
                .mc {{ border-radius:12px; overflow:hidden; border:1px solid #262626; }}
                .mc iframe {{ width:100%; height:200px; border:none; display:block; }}
                .ab {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
                .bp {{ display:flex; align-items:center; justify-content:center; gap:6px; padding:13px; border:none; border-radius:10px; background:linear-gradient(135deg,#3b82f6,#2563eb); color:#fff; font-size:13px; font-weight:600; text-decoration:none; }}
                .bs {{ display:flex; align-items:center; justify-content:center; gap:6px; padding:13px; border:1px solid #262626; border-radius:10px; background:#141414; color:#a3a3a3; font-size:13px; font-weight:600; text-decoration:none; }}
            </style></head>
            <body>
                <div class="w">
                    <div class="hd"><div class="si">✓</div><h1>Location Predicted</h1><p>Analysis completed for case {matched_user_id}</p></div>
                    <div class="rc">
                        <div class="rs"><div class="sl">Case Information</div>
                            <div class="dr"><span class="dk">Person ID</span><span class="dv">{matched_user_id}</span></div>
                            <div class="dr"><span class="dk">Target Date</span><span class="dv">{date}</span></div>
                            <div class="dr"><span class="dk">Target Time</span><span class="dv">{time}</span></div>
                        </div>
                        <div class="rs"><div class="sl">Predicted Coordinates</div>
                            <div class="cg">
                                <div class="cb"><div class="cl">Latitude</div><div class="cv">{lat}<span class="cu">° N</span></div></div>
                                <div class="cb"><div class="cl">Longitude</div><div class="cv">{lon}<span class="cu">° E</span></div></div>
                            </div>
                        </div>
                        <div class="rs"><div class="sl">Location Preview</div>
                            <div class="mc"><iframe src="https://maps.google.com/maps?q={lat},{lon}&hl=en&z=14&output=embed" loading="lazy"></iframe></div>
                        </div>
                    </div>
                    <div class="ab">
                        <a href="https://www.google.com/maps?q={lat},{lon}" target="_blank" class="bp">🗺️ Open in Maps</a>
                        <a href="/" class="bs">← New Prediction</a>
                    </div>
                </div>
            </body></html>
            """
    except Exception as e:
        err_msg = traceback.format_exc()
        return f"<h3>Internal Server Error</h3><pre>{err_msg}</pre>", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
