# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.inference_pipeline import generate_forecast

# ── Page config ──
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .aqi-number { font-size: 3rem; font-weight: bold; }
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("🌫️ Karachi AQI Forecasting Dashboard")
st.markdown("*Real-time Air Quality Index predictions for the next 3 days*")
st.divider()

# ── Load forecast ──
with st.spinner("🔮 Generating forecast..."):
    try:
        forecast = generate_forecast()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ── Current AQI ──
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #aaa;">Current AQI</div>
        <div class="aqi-number" style="color: {forecast['current_color']}">
            {forecast['current_aqi']}
        </div>
        <div style="color: {forecast['current_color']}; font-weight: bold;">
            {forecast['current_cat']}
        </div>
        <div style="color: #666; font-size: 0.8rem; margin-top: 8px;">
            {forecast['city']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #aaa;">Last Updated</div>
        <div style="font-size: 1.1rem; margin-top: 10px;">
            {forecast['generated_at']}
        </div>
        <div style="color: #aaa; font-size: 0.8rem; margin-top: 8px;">
            Auto-updates every hour
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Alert box
    aqi = forecast['current_aqi']
    if aqi > 300:
        st.markdown("""<div class="alert-box" style="background:#7e0023; color:white;">
        🚨 HAZARDOUS — Avoid all outdoor activities!</div>""", unsafe_allow_html=True)
    elif aqi > 200:
        st.markdown("""<div class="alert-box" style="background:#8f3f97; color:white;">
        ⚠️ VERY UNHEALTHY — Stay indoors if possible</div>""", unsafe_allow_html=True)
    elif aqi > 150:
        st.markdown("""<div class="alert-box" style="background:#ff0000; color:white;">
        ⚠️ UNHEALTHY — Limit prolonged outdoor exertion</div>""", unsafe_allow_html=True)
    elif aqi > 100:
        st.markdown("""<div class="alert-box" style="background:#ff7e00; color:white;">
        ⚠️ UNHEALTHY FOR SENSITIVE GROUPS — Take precautions</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="alert-box" style="background:#00e400; color:black;">
        ✅ AIR QUALITY IS ACCEPTABLE — Enjoy outdoor activities</div>""", unsafe_allow_html=True)

st.divider()

# ── 3-Day Daily Forecast ──
st.subheader("📅 3-Day Forecast")
daily = forecast["daily"][:3]
cols = st.columns(3)

for i, col in enumerate(cols):
    if i < len(daily):
        d = daily[i]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.2rem; font-weight: bold;">{d['date']}</div>
                <div class="aqi-number" style="color: {d['color']}; font-size: 2rem;">
                    {d['avg_aqi']:.0f}
                </div>
                <div style="color: {d['color']};">{d['category']}</div>
                <div style="color: #aaa; font-size: 0.8rem; margin-top: 8px;">
                    Min: {d['min_aqi']:.0f} | Max: {d['max_aqi']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ── Hourly Forecast Chart ──
st.subheader("📈 72-Hour AQI Forecast")
hourly_df = pd.DataFrame(forecast["hourly"])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hourly_df["datetime"],
    y=hourly_df["aqi"],
    mode="lines+markers",
    line=dict(color="#00bcd4", width=2),
    marker=dict(size=4),
    name="Predicted AQI",
    fill="tozeroy",
    fillcolor="rgba(0,188,212,0.1)"
))

# Add AQI threshold lines
thresholds = [(50, "Good", "#00e400"), (100, "Moderate", "#ffff00"),
              (150, "USG", "#ff7e00"), (200, "Unhealthy", "#ff0000")]
for val, label, color in thresholds:
    fig.add_hline(y=val, line_dash="dash", line_color=color,
                  annotation_text=label, annotation_position="right")

fig.update_layout(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1e2130",
    font=dict(color="white"),
    xaxis=dict(showgrid=False, title="Date & Time"),
    yaxis=dict(showgrid=True, gridcolor="#2d3250", title="AQI"),
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ── SHAP Feature Importance ──
st.subheader("🔍 Feature Importance (SHAP)")
shap_path = "models/shap_importance.csv"
if os.path.exists(shap_path):
    shap_df = pd.read_csv(shap_path).head(10)
    fig2 = px.bar(
        shap_df, x="importance", y="feature",
        orientation="h", color="importance",
        color_continuous_scale="Blues",
        title="Top 10 Most Important Features"
    )
    fig2.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1e2130",
        font=dict(color="white"),
        height=400,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── AQI Scale Reference ──
st.subheader("📊 AQI Scale Reference")
scale_data = {
    "Category":   ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"],
    "AQI Range":  ["0-50", "51-100", "101-150", "151-200", "201-300", "300+"],
    "Color":      ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"],
}
st.dataframe(pd.DataFrame(scale_data), use_container_width=True, hide_index=True)

st.caption(f"Data source: AQICN API | Model: Ridge Regression (R²=0.9994) | Updated: {forecast['generated_at']}")