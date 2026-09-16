import streamlit as st
import pandas as pd
from pathlib import Path

# ==============================
# CUSTOM CSS DESIGN
# ==============================

st.markdown("""
<style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #eef4ff 0%,
            #f8fafc 50%,
            #e0f2fe 100%
        );
    }

    /* Main content */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }

    /* Section headings */
    h2 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        border-left: 6px solid #2563eb;
        padding-left: 12px;
        margin-top: 25px;
    }

    h3 {
        color: #1e40af !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #dbeafe;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.10);
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #1d4ed8 !important;
        font-weight: 800;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a,
            #1e3a8a
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Select box */
    [data-baseweb="select"] > div {
        background-color: white;
        border-radius: 10px;
    }

    /* Information boxes */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        border-radius: 10px;
    }

    /* Data table */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.08);
    }

    /* Map */
    iframe {
        border-radius: 15px;
    }

    /* Text */
    p {
        color: #334155;
    }

</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent

# Page settings
st.set_page_config(
    page_title="CASEFILE - Missing Person Investigation",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ CASEFILE")
st.subheader("AI-Powered Missing Person Investigation and Location Prediction System")

st.info(
    "Academic simulation using synthetic/public GPS data. "
    "Predictions are probabilistic and should not be treated as proof of a person's location."
)

# Load data
data_file = BASE_DIR / "data" / "priority_scored_gps.csv"

if not data_file.exists():
    st.error("Priority scored dataset was not found.")
    st.stop()

df = pd.read_csv(data_file)

# Sidebar
st.sidebar.header("Case Selection")

user_ids = sorted(df["User_ID"].unique())

selected_user = st.sidebar.selectbox(
    "Select User ID",
    user_ids
)

user_data = df[df["User_ID"] == selected_user]

# Case information
st.header("Case Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("User ID", selected_user)

with col2:
    st.metric("Records", len(user_data))

with col3:
    st.metric(
        "Priority",
        user_data["Priority_Level"].mode()[0]
    )

# Latest location
latest = user_data.iloc[-1]

st.header("Latest Location")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latitude",
        f"{latest['Latitude']:.6f}"
    )

with col2:
    st.metric(
        "Longitude",
        f"{latest['Longitude']:.6f}"
    )

# Priority score
st.header("Priority Score")

st.progress(
    int(min(max(latest["Priority_Score"], 0), 100))
)

st.write(
    f"Priority Score: **{latest['Priority_Score']:.2f} / 100**"
)

# Anomaly information
st.header("Movement Analysis")

anomaly_count = int(
    (user_data["Anomaly"] == -1).sum()
)

normal_count = int(
    (user_data["Anomaly"] == 1).sum()
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Normal Records", normal_count)

with col2:
    st.metric("Anomaly Records", anomaly_count)

# GPS map
st.header("GPS Movement Map")

map_data = user_data[["Latitude", "Longitude"]].copy()

map_data["Latitude"] = pd.to_numeric(
    map_data["Latitude"],
    errors="coerce"
)

map_data["Longitude"] = pd.to_numeric(
    map_data["Longitude"],
    errors="coerce"
)

map_data = map_data.dropna()

if not map_data.empty:
    st.map(
        map_data,
        latitude="Latitude",
        longitude="Longitude"
    )
else:
    st.warning("No valid GPS coordinates available.")

# Data table
st.header("Case Data")

st.dataframe(
    user_data[
        [
            "User_ID",
            "Latitude",
            "Longitude",
            "Cluster",
            "Anomaly_Label",
            "Priority_Score",
            "Priority_Level"
        ]
    ].tail(20),
    use_container_width=True
)

st.caption(
    "This dashboard is for academic demonstration only. "
    "Model outputs are estimates and may contain errors or false positives."
)
