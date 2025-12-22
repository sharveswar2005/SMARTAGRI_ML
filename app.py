import streamlit as st
import requests

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://localhost:8000/predict"

# Persistent HTTP session
session = requests.Session()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SmartAgriML | Crop Failure Risk",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# CUSTOM CSS (Dark, Clean UI)
# =========================================================
st.markdown("""
<style>
section.main > div {
    background-color: transparent;
}

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #e5e7eb;
}

.subtitle {
    font-size: 18px;
    color: #c7d2fe;
    margin-bottom: 40px;
}

.card {
    background: #020617;
    padding: 28px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 15px 35px rgba(0,0,0,0.7);
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    padding: 12px 26px;
    font-weight: 600;
    border: none;
}

.risk-high {
    background: linear-gradient(90deg, #dc2626, #ef4444);
    color: white;
    padding: 14px;
    border-radius: 12px;
    font-weight: bold;
    text-align: center;
}

.risk-medium {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
    color: black;
    padding: 14px;
    border-radius: 12px;
    font-weight: bold;
    text-align: center;
}

.risk-low {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    padding: 14px;
    border-radius: 12px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🌾 SmartAgriML</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered Crop Failure Risk Prediction System</div>',
    unsafe_allow_html=True
)

# =========================================================
# LAYOUT
# =========================================================
col1, col2 = st.columns([1.25, 1])

# =========================================================
# INPUT CARD
# =========================================================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Enter Crop & Climate Details")

    Area = st.number_input("Area (hectares)", min_value=0.0, value=1200.0)
    Item = st.text_input("Crop Name", value="Rice")
    Year = st.number_input("Year", min_value=1900, max_value=2100, value=2018)
    Rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1200.0)
    Pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=1.2)
    Temperature = st.number_input("Average Temperature (°C)", min_value=-10.0, value=26.0)

    predict_btn = st.button("🚀 Predict Risk")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# OUTPUT CARD
# =========================================================
# =========================================================
# OUTPUT CARD
# =========================================================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    output_placeholder = st.empty()

    if predict_btn:
        payload = {
            "Area": Area,
            "Item": Item,
            "Year": Year,
            "average_rain_fall_mm_per_year": Rainfall,
            "pesticides_tonnes": Pesticides,
            "avg_temp": Temperature
        }

        try:
            # Step 1: Analyzing text
            output_placeholder.markdown(
                "🔍 **Analyzing crop & climate data...**"
            )

            # Small delay for UX feel (optional but nice)
            import time
            time.sleep(0.6)

            # Step 2: Model execution text
            output_placeholder.markdown(
                "📡 **Running machine learning risk model...**"
            )

            # Actual backend call
            response = session.post(API_URL, json=payload)

            result = response.json()
            score = result["crop_failure_risk_score"]
            level = result["risk_level"]

            # Step 3: Final output
            output_placeholder.empty()
            st.metric("Risk Score", f"{score:.2f}")

            if level == "High":
                st.markdown('<div class="risk-high">HIGH RISK</div>', unsafe_allow_html=True)
            elif level == "Medium":
                st.markdown('<div class="risk-medium">MEDIUM RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">LOW RISK</div>', unsafe_allow_html=True)

        except Exception:
            output_placeholder.empty()
            st.error("Backend is not running. Start FastAPI first.")

    else:
        st.info("Enter details and click **Predict Risk**")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    "<hr><center style='color:#9ca3af;'>© 2025 SmartAgriML | Machine Learning Portfolio Project</center>",
    unsafe_allow_html=True
)
