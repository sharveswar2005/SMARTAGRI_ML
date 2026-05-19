import streamlit as st
import requests
import pandas as pd
import time
import altair as alt

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://localhost:8000/predict"
session = requests.Session()

st.set_page_config(
    page_title="SmartAgriML | Crop Failure Risk",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'history' not in st.session_state:
    st.session_state.history = []

# =========================================================
# CUSTOM CSS (Premium UI)
# =========================================================
st.markdown("""
<style>
.stApp { background: #0f172a; color: #f8fafc; }
.main-title { font-size: 42px; font-weight: 800; color: #e5e7eb; text-align: center; }
.subtitle { font-size: 16px; color: #94a3b8; text-align: center; margin-bottom: 30px; }

/* Gradient Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    text-align: center;
}
.metric-value { font-size: 36px; font-weight: bold; color: #38bdf8; }
.metric-label { font-size: 14px; text-transform: uppercase; color: #64748b; font-weight: 600; letter-spacing: 1px; }

.risk-High { border-top: 4px solid #ef4444; }
.risk-High .metric-value { color: #ef4444; }
.risk-Medium { border-top: 4px solid #f59e0b; }
.risk-Medium .metric-value { color: #f59e0b; }
.risk-Low { border-top: 4px solid #10b981; }
.risk-Low .metric-value { color: #10b981; }

hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🌾 SmartAgriML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Prediction of Crop Failure Risk & Yield Security</div><hr>', unsafe_allow_html=True)

# =========================================================
# SIDEBAR (INPUTS)
# =========================================================
with st.sidebar:
    st.title("⚙️ Parameters")
    st.markdown("Enter environmental and crop details:")
    
    Item = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Soybeans", "Potatoes"], index=0)
    Area = st.number_input("Area (hectares)", min_value=1.0, value=1200.0, step=100.0)
    Year = st.slider("Year", min_value=2000, max_value=2030, value=2025)
    Rainfall = st.slider("Rainfall (mm/yr)", min_value=0.0, max_value=4000.0, value=1200.0)
    Temperature = st.slider("Temperature (°C)", min_value=-5.0, max_value=50.0, value=26.0)
    Pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=1.2, step=0.1)

    predict_btn = st.button("🚀 Predict Risk Level", use_container_width=True, type="primary")

# =========================================================
# MAIN CONTENT
# =========================================================
if predict_btn:
    payload = {
        "Area": Area,
        "Item": Item,
        "Year": Year,
        "average_rain_fall_mm_per_year": Rainfall,
        "pesticides_tonnes": Pesticides,
        "avg_temp": Temperature
    }

    with st.spinner("Analyzing parameters and computing SHAP factors..."):
        time.sleep(0.8) # UX enhancement
        try:
            response = session.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                
                score = result.get("crop_failure_risk_score", 0)
                level = result.get("risk_level", "Unknown")
                conf = result.get("confidence", 0.0) * 100
                factors = result.get("key_factors", [])

                # Save to history
                st.session_state.history.insert(0, {
                    "Crop": Item, "Area (ha)": Area, "Rainfall": Rainfall, 
                    "Temp (°C)": Temperature, "Risk Score": f"{score:.1f}", "Level": level
                })
                
                # --- KPI CARDS ---
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'<div class="metric-card risk-{level}"><div class="metric-label">Risk Category</div><div class="metric-value">{level.upper()}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Score (0-100)</div><div class="metric-value">{score:.1f}</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Model Confidence</div><div class="metric-value">{conf:.1f}%</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- CHARTS ---
                col_chart1, col_chart2 = st.columns([1.2, 1])
                
                with col_chart1:
                    st.subheader("🔍 Feature Importance (Explainability)")
                    if factors:
                        factor_df = pd.DataFrame(factors)
                        # Normalize importance for better UI chart scaling
                        max_imp = factor_df['importance'].max()
                        factor_df['importance'] = (factor_df['importance'] / max_imp) * 100
                        
                        chart = alt.Chart(factor_df).mark_bar(cornerRadiusEnd=4).encode(
                            x=alt.X('importance:Q', title='Relative Importance (%)'),
                            y=alt.Y('feature:N', sort='-x', title=''),
                            color=alt.condition(
                                alt.datum.importance > 50,
                                alt.value('#ef4444'),  # High imp -> Red
                                alt.value('#64748b')   # Low imp -> Slate
                            ),
                            tooltip=['feature', 'importance']
                        ).properties(height=250)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Feature importances not available for this model.")
                        
                with col_chart2:
                    st.subheader("💡 Analysis Summary")
                    st.markdown(f"""
                    - **Primary Issue**: The most critical driver of this risk score is **{factors[0]['feature'] if factors else 'Unknown'}**.
                    - **Temperature Impact**: At {Temperature}°C, crop resilience changes strictly depending on rainfall.
                    - **Pesticide Usage**: {Pesticides} tonnes applied to {Area} hectares.
                    """)
                    if level == "High":
                        st.warning("Immediate intervention required to mitigate expected yield losses.")
                    elif level == "Medium":
                        st.info("Monitor weather and consider preemptive defensive agricultural practices.")
                    else:
                        st.success("Conditions are currently optimal for maximum yield.")

            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to Backend API. Ensure FastAPI is running on port 8000. Error: {str(e)}")

else:
    st.info("👈 Adjust the parameters in the sidebar and click **Predict Risk Level** to start.")

# =========================================================
# PREDICTION HISTORY
# =========================================================
if st.session_state.history:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("🕒 Recent Predictions Session History")
    st.dataframe(pd.DataFrame(st.session_state.history).head(5), use_container_width=True)

