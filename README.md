# 🕵️ CASEFILE - Missing Person Location Prediction System

An AI-powered machine learning web application built with **Flask** and **Scikit-Learn** to predict probable missing person locations based on historical GPS movement data.

---

## 🚀 Features

- **Location Prediction**: Predicts approximate Latitude and Longitude using RandomForestRegressor.
- **Auto-Training Fallback**: Auto-generates model artifacts (`location_model.pkl`, `user_encoder.pkl`) on startup if pickle files are missing or empty.
- **Web UI & Result Page**: Beautiful glassmorphism UI with direct Google Maps location link.
- **Deployment Ready**: Fully configured for **Render**, **Vercel**, **Docker**, **Railway**, and local deployment.

---

## 🛠️ Project Structure

```text
├── app.py              # Main Flask application with auto-training fallback
├── train_model.py      # Script to train RandomForest model on GPS dataset
├── predict.py          # Interactive CLI location predictor
├── test_app.py         # Automated test suite for Flask endpoints
├── cleaned_gps_data.csv# Primary GPS movement dataset
├── location_model.pkl  # Trained ML model artifact
├── user_encoder.pkl    # LabelEncoder artifact
├── Procfile            # Deployment process file (gunicorn)
├── vercel.json         # Vercel serverless deployment config
├── Dockerfile          # Container build instructions
├── docker-compose.yml  # Docker Compose orchestration
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # Main CSS stylesheet
└── templates/
    ├── index.html      # Main prediction form UI
    └── result.html     # Prediction result page
```

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/NithishEerasarapu/CASEFILE-Missing-Person-Location-Prediction.git
cd CASEFILE-Missing-Person-Location-Prediction
```

### 2. Set up virtual environment & install requirements
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 🌐 Deploying to Cloud Platforms

### 1. Deploying to **Render** (Recommended - FREE)
1. Push code to your GitHub repository.
2. Sign in to [Render](https://render.com/).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Set the settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click **Create Web Service**. Done!

---

### 2. Deploying to **Vercel**
1. Install Vercel CLI or connect GitHub on [vercel.com](https://vercel.com).
2. Push your project to GitHub.
3. Import the repo on Vercel. Vercel automatically detects `vercel.json` and deploys `app.py`.

---

### 3. Deploying with **Docker**
```bash
# Build & start container
docker-compose up --build
```
Access at `http://localhost:5000`.

---

## 🧪 Running Automated Tests
```bash
python test_app.py
```
All unit tests should pass with `OK`.

---

## 📄 License
Academic Machine Learning Demonstration © 2026