# app/streamlit_app.py
# Production-grade AQI Forecasting Dashboard for Karachi

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
import joblib
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines"))

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark industrial aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #080c14;
    --surface:   #0d1421;
    --surface2:  #131d2e;
    --border:    #1e2d42;
    --accent:    #00d4ff;
    --accent2:   #ff6b35;
    --good:      #00e676;
    --moderate:  #ffea00;
    --sensitive: #ff9100;
    --unhealthy: #ff1744;
    --vunhealthy:#d500f9;
    --hazardous: #ff006e;
    --text:      #e8f4f8;
    --muted:     #5a7a8a;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }

.aqi-hero {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.aqi-number {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.alert-critical {
    background: linear-gradient(135deg, #7e0023, #ff006e);
    border-radius: 12px;
    padding: 16px 20px;
    font-weight: 600;
    animation: pulse 2s infinite;
}
.alert-warning {
    background: linear-gradient(135deg, #3d1a00, #ff6b35);
    border-radius: 12px;
    padding: 16px 20px;
    font-weight: 600;
}
.alert-ok {
    background: linear-gradient(135deg, #002213, #00e676);
    border-radius: 12px;
    padding: 16px 20px;
    color: #002213;
    font-weight: 600;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

.forecast-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

div[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--surface2) !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
AQI_COLORS = {
    "Good":                              "#00e676",
    "Moderate":                          "#ffea00",
    "Unhealthy for Sensitive Groups":    "#ff9100",
    "Unhealthy":                         "#ff1744",
    "Very Unhealthy":                    "#d500f9",
    "Hazardous":                         "#ff006e",
}

def aqi_category(aqi):
    if aqi <= 50:    return "Good",                           "#00e676"
    elif aqi <= 100: return "Moderate",                       "#ffea00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff9100"
    elif aqi <= 200: return "Unhealthy",                      "#ff1744"
    elif aqi <= 300: return "Very Unhealthy",                 "#d500f9"
    else:            return "Hazardous",                      "#ff006e"

def plotly_layout(title="", height=400):
    return dict(
        title=title,
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1421",
        font=dict(color="#e8f4f8", family="DM Sans"),
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=False, color="#5a7a8a"),
        yaxis=dict(showgrid=True, gridcolor="#1e2d42", color="#5a7a8a"),
    )

@st.cache_data(ttl=3600)
def load_forecast():
    from pipelines.inference_pipeline import generate_forecast
    return generate_forecast()

@st.cache_data(ttl=3600)
def load_historical_data():
    """Load data from Supabase or CSV fallback"""
    try:
        from supabase import create_client
        from dotenv import load_dotenv
        load_dotenv()
        supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        all_rows, offset = [], 0
        while True:
            result = (supabase.table("aqi_features")
                      .select("*").order("timestamp")
                      .range(offset, offset + 999).execute())
            if not result.data: break
            all_rows.extend(result.data)
            offset += 1000
            if len(result.data) < 1000: break
        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.warning(f"Using CSV fallback: {e}")
        df = pd.read_csv("data/aqi_features_rows.csv")
        df = df.rename(columns={"temperature": "temp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=86400)
def load_model_metrics():
    metrics_path = "models/metrics.json"
    shap_path    = "models/shap_importance.csv"
    name_path    = "models/best_model_name.json"
    metrics, shap_df, model_name = {}, None, "Unknown"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f: metrics = json.load(f)
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
    if os.path.exists(name_path):
        with open(name_path) as f: model_name = json.load(f)["name"]
    return metrics, shap_df, model_name

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-family: Space Mono; font-size: 1.4rem; color: #00d4ff;'>
            🌫️ AQI WATCH
        </div>
        <div style='color: #5a7a8a; font-size: 0.8rem; margin-top: 4px;'>
            Karachi Air Quality Monitor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.selectbox(
        "Navigate",
        ["📊 Live Dashboard", "📈 EDA & Trends", "🔍 Model Insights", "⚠️ AQI Guide"],
        label_visibility="collapsed"
    )

    st.divider()

    # AQI Scale
    st.markdown("<div class='section-header'>AQI Scale</div>", unsafe_allow_html=True)
    scale = [
        ("0–50",   "Good",              "#00e676"),
        ("51–100", "Moderate",          "#ffea00"),
        ("101–150","Sensitive Groups",  "#ff9100"),
        ("151–200","Unhealthy",         "#ff1744"),
        ("201–300","Very Unhealthy",    "#d500f9"),
        ("300+",   "Hazardous",         "#ff006e"),
    ]
    for rng, label, color in scale:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin:6px 0;'>
            <div style='width:12px; height:12px; border-radius:50%; background:{color}; flex-shrink:0;'></div>
            <span style='color:#5a7a8a; font-size:0.75rem;'>{rng}</span>
            <span style='font-size:0.75rem;'>{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Loading forecast data..."):
    try:
        forecast    = load_forecast()
        hist_df     = load_historical_data()
        metrics, shap_df, model_name = load_model_metrics()
        data_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        data_loaded = False
        st.stop()

# ═══════════════════════════════════════════
# PAGE 1 — LIVE DASHBOARD
# ═══════════════════════════════════════════
if page == "📊 Live Dashboard":

    st.markdown("""
    <div style='font-family: Space Mono; font-size: 0.7rem; color: #5a7a8a;
                letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 4px;'>
        Real-Time Monitor
    </div>
    <h1 style='margin: 0; font-size: 2rem;'>Karachi Air Quality</h1>
    """, unsafe_allow_html=True)

    current_aqi   = forecast["current_aqi"]
    cat, color    = aqi_category(current_aqi)
    generated_at  = forecast.get("generated_at", "")

    # ── Alert Banner ──
    if current_aqi > 300:
        st.markdown(f"""<div class='alert-critical'>
        🚨 HAZARDOUS AIR QUALITY — AQI {current_aqi} — Stay indoors. Avoid ALL outdoor activity.
        </div>""", unsafe_allow_html=True)
    elif current_aqi > 200:
        st.markdown(f"""<div class='alert-warning'>
        ⚠️ VERY UNHEALTHY — AQI {current_aqi} — Sensitive groups must stay indoors.
        </div>""", unsafe_allow_html=True)
    elif current_aqi > 150:
        st.markdown(f"""<div class='alert-warning'>
        ⚠️ UNHEALTHY — AQI {current_aqi} — Limit prolonged outdoor exertion.
        </div>""", unsafe_allow_html=True)
    elif current_aqi > 100:
        st.markdown(f"""<div class='alert-warning'>
        ⚠️ SENSITIVE GROUPS AT RISK — AQI {current_aqi} — Children and elderly should stay indoors.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='alert-ok'>
        ✅ AIR QUALITY ACCEPTABLE — AQI {current_aqi} — Enjoy outdoor activities.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Current AQI + Stats ──
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class='aqi-hero'>
            <div style='color:#5a7a8a; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase;'>
                Current AQI
            </div>
            <div class='aqi-number' style='color:{color};'>{current_aqi}</div>
            <div style='color:{color}; font-weight:600; font-size:1rem;'>{cat}</div>
            <div style='color:#5a7a8a; font-size:0.75rem; margin-top:12px;'>
                📍 Karachi, Pakistan<br>
                🕐 {generated_at}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Pollutants</div>", unsafe_allow_html=True)
        pm25 = forecast.get("current_pm25", 0)
        pm10 = forecast.get("current_pm10", 0)
        o3   = forecast.get("current_o3", 0)
        no2  = forecast.get("current_no2", 0)
        for label, val, unit in [
            ("PM2.5", pm25, "μg/m³"),
            ("PM10",  pm10, "μg/m³"),
            ("O₃",    o3,   "μg/m³"),
            ("NO₂",   no2,  "μg/m³"),
        ]:
            st.markdown(f"""
            <div class='stat-row'>
                <span style='color:#5a7a8a;'>{label}</span>
                <span style='font-family: Space Mono;'>{val:.1f} {unit}</span>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='section-header'>Weather</div>", unsafe_allow_html=True)
        temp     = forecast.get("current_temp", 0)
        humidity = forecast.get("current_humidity", 0)
        wind     = forecast.get("current_wind", 0)
        pressure = forecast.get("current_pressure", 0)
        for label, val, unit in [
            ("Temperature", temp,     "°C"),
            ("Humidity",    humidity, "%"),
            ("Wind Speed",  wind,     "m/s"),
            ("Pressure",    pressure, "hPa"),
        ]:
            st.markdown(f"""
            <div class='stat-row'>
                <span style='color:#5a7a8a;'>{label}</span>
                <span style='font-family: Space Mono;'>{val:.1f} {unit}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3-Day Daily Forecast Cards ──
    st.markdown("<div class='section-header'>3-Day Forecast</div>", unsafe_allow_html=True)
    daily = forecast.get("daily", [])[:3]
    cols  = st.columns(3)

    for i, col in enumerate(cols):
        if i < len(daily):
            d = daily[i]
            cat_d, color_d = aqi_category(d["avg_aqi"])
            with col:
                st.markdown(f"""
                <div class='forecast-card'>
                    <div style='font-family: Space Mono; font-size:1rem; color:#5a7a8a;'>{d['date']}</div>
                    <div style='font-family: Space Mono; font-size:2.5rem; font-weight:700;
                                color:{color_d}; margin:8px 0;'>{d['avg_aqi']:.0f}</div>
                    <div style='color:{color_d}; font-size:0.85rem; font-weight:600;'>{cat_d}</div>
                    <div style='margin-top:12px; color:#5a7a8a; font-size:0.8rem;'>
                        <span>↓ {d['min_aqi']:.0f}</span>
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        <span>↑ {d['max_aqi']:.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 72-Hour Hourly Chart ──
    st.markdown("<div class='section-header'>72-Hour Forecast</div>", unsafe_allow_html=True)
    hourly_df = pd.DataFrame(forecast.get("hourly", []))

    if not hourly_df.empty:
        fig = go.Figure()

        # Color zones
        for y0, y1, clr, name in [
            (0,   50,  "rgba(0,230,118,0.05)",  "Good"),
            (50,  100, "rgba(255,234,0,0.05)",   "Moderate"),
            (100, 150, "rgba(255,145,0,0.05)",   "Sensitive"),
            (150, 200, "rgba(255,23,68,0.05)",   "Unhealthy"),
            (200, 300, "rgba(213,0,249,0.05)",   "Very Unhealthy"),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, line_width=0)

        # Threshold lines
        for val, clr in [(50,"#00e676"),(100,"#ffea00"),(150,"#ff9100"),(200,"#ff1744"),(300,"#d500f9")]:
            fig.add_hline(y=val, line_dash="dot", line_color=clr, line_width=1, opacity=0.5)

        fig.add_trace(go.Scatter(
            x=hourly_df["datetime"],
            y=hourly_df["aqi"],
            mode="lines",
            line=dict(color="#00d4ff", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.08)",
            name="Predicted AQI",
            hovertemplate="<b>%{x}</b><br>AQI: %{y}<extra></extra>"
        ))

        layout = plotly_layout(height=380)
        layout["xaxis"]["title"] = "Date & Hour"
        layout["yaxis"]["title"] = "AQI"
        layout["yaxis"]["range"] = [0, max(300, hourly_df["aqi"].max() + 20)]
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # ── Model Info ──
    if metrics:
        st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",  model_name)
        c2.metric("RMSE",        f"{metrics.get('rmse', 0):.2f}")
        c3.metric("MAE",         f"{metrics.get('mae', 0):.2f}")
        c4.metric("R² Score",    f"{metrics.get('r2', 0):.4f}")

# ═══════════════════════════════════════════
# PAGE 2 — EDA & TRENDS
# ═══════════════════════════════════════════
elif page == "📈 EDA & Trends":
    st.markdown("<h1 style='margin-bottom:4px;'>EDA & Trends</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a8a;'>Exploratory analysis of historical Karachi AQI data</p>", unsafe_allow_html=True)

    df = hist_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None) if df["timestamp"].dt.tz is not None else df["timestamp"]
    df = df.dropna(subset=["timestamp", "aqi"])

    # ── Summary Stats ──
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Avg AQI",        f"{df['aqi'].mean():.1f}")
    c3.metric("Max AQI",        f"{df['aqi'].max():.0f}")
    c4.metric("Min AQI",        f"{df['aqi'].min():.0f}")
    c5.metric("Date Range",     f"{len(df['timestamp'].dt.date.unique())} days")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AQI Over Time ──
    st.markdown("<div class='section-header'>AQI Over Time</div>", unsafe_allow_html=True)
    df_sorted = df.sort_values("timestamp")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_sorted["timestamp"], y=df_sorted["aqi"],
        mode="lines",
        line=dict(color="#00d4ff", width=1),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        name="AQI",
        hovertemplate="%{x}<br>AQI: %{y:.1f}<extra></extra>"
    ))
    # 7-day rolling average
    df_sorted["aqi_7d"] = df_sorted["aqi"].rolling(168).mean()
    fig1.add_trace(go.Scatter(
        x=df_sorted["timestamp"], y=df_sorted["aqi_7d"],
        mode="lines",
        line=dict(color="#ff6b35", width=2.5, dash="solid"),
        name="7-Day Avg",
        hovertemplate="%{x}<br>7d Avg: %{y:.1f}<extra></extra>"
    ))
    fig1.update_layout(**plotly_layout(height=350))
    fig1.update_layout(legend=dict(bgcolor="#0d1421", bordercolor="#1e2d42"))
    st.plotly_chart(fig1, use_container_width=True)

    # ── AQI Distribution + Monthly Avg ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>AQI Distribution</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=df["aqi"], nbinsx=40,
            marker=dict(
                color=df["aqi"].values,
                colorscale=[[0,"#00e676"],[0.25,"#ffea00"],[0.5,"#ff9100"],[0.75,"#ff1744"],[1,"#ff006e"]],
                cmin=0, cmax=300,
                line=dict(width=0)
            ),
            name="AQI Distribution"
        ))
        fig2.update_layout(**plotly_layout(height=320))
        fig2.update_layout(xaxis_title="AQI", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Monthly Average AQI</div>", unsafe_allow_html=True)
        df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
        df["month_num"]  = df["timestamp"].dt.to_period("M")
        monthly = df.groupby("month_num")["aqi"].mean().reset_index()
        monthly["month_str"] = monthly["month_num"].astype(str)
        monthly["color"]     = monthly["aqi"].apply(lambda x: aqi_category(x)[1])

        fig3 = go.Figure(go.Bar(
            x=monthly["month_str"],
            y=monthly["aqi"],
            marker_color=monthly["color"],
            text=monthly["aqi"].round(1),
            textposition="outside",
            textfont=dict(color="#e8f4f8", size=11)
        ))
        fig3.update_layout(**plotly_layout(height=320))
        fig3.update_layout(xaxis_title="Month", yaxis_title="Avg AQI")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Hourly Pattern + Day of Week ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-header'>Average AQI by Hour</div>", unsafe_allow_html=True)
        df["hour"] = pd.to_numeric(df.get("hour", df["timestamp"].dt.hour), errors="coerce")
        hourly_avg = df.groupby("hour")["aqi"].mean().reset_index()
        fig4 = go.Figure(go.Bar(
            x=hourly_avg["hour"],
            y=hourly_avg["aqi"],
            marker=dict(
                color=hourly_avg["aqi"],
                colorscale=[[0,"#00d4ff"],[0.5,"#ff6b35"],[1,"#ff006e"]],
                cmin=hourly_avg["aqi"].min(),
                cmax=hourly_avg["aqi"].max(),
            ),
            hovertemplate="Hour %{x}:00<br>Avg AQI: %{y:.1f}<extra></extra>"
        ))
        fig4.update_layout(**plotly_layout(height=300))
        fig4.update_layout(xaxis_title="Hour of Day", yaxis_title="Avg AQI",
                           xaxis=dict(tickmode="linear", dtick=4))
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Average AQI by Day of Week</div>", unsafe_allow_html=True)
        df["dow"] = df["timestamp"].dt.dayofweek
        dow_avg   = df.groupby("dow")["aqi"].mean().reset_index()
        days      = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow_avg["day_name"] = dow_avg["dow"].apply(lambda x: days[x])
        fig5 = go.Figure(go.Bar(
            x=dow_avg["day_name"],
            y=dow_avg["aqi"],
            marker=dict(
                color=dow_avg["aqi"],
                colorscale=[[0,"#00d4ff"],[1,"#ff006e"]],
            ),
            hovertemplate="%{x}<br>Avg AQI: %{y:.1f}<extra></extra>"
        ))
        fig5.update_layout(**plotly_layout(height=300))
        fig5.update_layout(xaxis_title="Day", yaxis_title="Avg AQI")
        st.plotly_chart(fig5, use_container_width=True)

    # ── Pollutant Correlations ──
    st.markdown("<div class='section-header'>Pollutant Correlation with AQI</div>", unsafe_allow_html=True)
    pollutants = ["pm25","pm10","o3","no2","co","so2"]
    available  = [p for p in pollutants if p in df.columns]

    if available:
        corr_vals = [df[p].corr(df["aqi"]) for p in available]
        colors    = ["#00d4ff" if c >= 0 else "#ff6b35" for c in corr_vals]
        fig6 = go.Figure(go.Bar(
            x=available, y=corr_vals,
            marker_color=colors,
            text=[f"{c:.3f}" for c in corr_vals],
            textposition="outside",
            textfont=dict(color="#e8f4f8")
        ))
        fig6.update_layout(**plotly_layout(height=300))
        fig6.update_layout(
            xaxis_title="Pollutant",
            yaxis_title="Correlation with AQI",
            yaxis_range=[-1, 1]
        )
        fig6.add_hline(y=0, line_color="#5a7a8a", line_width=1)
        st.plotly_chart(fig6, use_container_width=True)

    # ── AQI Category Breakdown ──
    st.markdown("<div class='section-header'>AQI Category Breakdown</div>", unsafe_allow_html=True)
    bins   = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good","Moderate","Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
    df["aqi_cat"] = pd.cut(df["aqi"], bins=bins, labels=labels)
    cat_counts    = df["aqi_cat"].value_counts().reindex(labels, fill_value=0)
    cat_colors    = ["#00e676","#ffea00","#ff9100","#ff1744","#d500f9","#ff006e"]

    col5, col6 = st.columns(2)
    with col5:
        fig7 = go.Figure(go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            marker_colors=cat_colors,
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(color="#e8f4f8")
        ))
        fig7.update_layout(
            paper_bgcolor="#080c14",
            font=dict(color="#e8f4f8"),
            height=320,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        fig8 = go.Figure(go.Bar(
            x=cat_counts.index,
            y=cat_counts.values,
            marker_color=cat_colors,
            text=cat_counts.values,
            textposition="outside",
            textfont=dict(color="#e8f4f8")
        ))
        fig8.update_layout(**plotly_layout(height=320))
        fig8.update_layout(xaxis_title="Category", yaxis_title="Hours")
        st.plotly_chart(fig8, use_container_width=True)

# ═══════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════
elif page == "🔍 Model Insights":
    st.markdown("<h1 style='margin-bottom:4px;'>Model Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a8a;'>SHAP explainability and model performance analysis</p>", unsafe_allow_html=True)

    # ── Model Metrics ──
    st.markdown("<div class='section-header'>Best Model Performance</div>", unsafe_allow_html=True)
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model",   model_name)
        c2.metric("RMSE ↓",  f"{metrics.get('rmse', 0):.2f}",  delta="Lower is better",  delta_color="off")
        c3.metric("MAE ↓",   f"{metrics.get('mae', 0):.2f}",   delta="Lower is better",  delta_color="off")
        c4.metric("R² ↑",    f"{metrics.get('r2', 0):.4f}",   delta="Higher is better", delta_color="off")
    else:
        st.warning("No model metrics found. Run training pipeline first.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP Feature Importance ──
    st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
    if shap_df is not None and not shap_df.empty:
        top_shap = shap_df.head(15)
        fig_shap = go.Figure(go.Bar(
            x=top_shap["importance"],
            y=top_shap["feature"],
            orientation="h",
            marker=dict(
                color=top_shap["importance"],
                colorscale=[[0,"#00d4ff"],[0.5,"#ff6b35"],[1,"#ff006e"]],
                cmin=0,
                cmax=top_shap["importance"].max(),
            ),
            text=top_shap["importance"].round(3),
            textposition="outside",
            textfont=dict(color="#e8f4f8")
        ))
        layout = plotly_layout(height=480)
        layout["xaxis"]["title"] = "Mean |SHAP Value|"
        layout["yaxis"]["title"] = "Feature"
        layout["yaxis"]["autorange"] = "reversed"
        fig_shap.update_layout(**layout)
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("<div class='section-header'>Feature Importance Table</div>", unsafe_allow_html=True)
        st.dataframe(
            shap_df.rename(columns={"feature": "Feature", "importance": "SHAP Importance"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("SHAP values not found. Run training pipeline with SHAP enabled.")

    # ── How the model works ──
    st.markdown("<div class='section-header'>How the Model Works</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1421; border:1px solid #1e2d42; border-radius:12px; padding:20px;'>
        <p>The AQI forecasting model uses a <b>multi-step recursive forecasting</b> approach:</p>
        <ol style='color:#e8f4f8; line-height:2;'>
            <li>Fetches current real-time AQI and weather data from OpenWeather API</li>
            <li>Engineers lag features (previous 1, 2, 3 hours AQI) and rolling averages</li>
            <li>Feeds features into the trained model to predict next hour's AQI</li>
            <li>Uses predicted AQI as input for the following hour (recursive)</li>
            <li>Repeats for 72 hours to generate 3-day forecast</li>
        </ol>
        <p style='color:#5a7a8a; font-size:0.85rem; margin-top:12px;'>
        ⚠️ Note: Recursive forecasting can accumulate errors over time.
        Predictions beyond 24 hours should be treated as trend estimates.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# PAGE 4 — AQI GUIDE
# ═══════════════════════════════════════════
elif page == "⚠️ AQI Guide":
    st.markdown("<h1>AQI Health Guide</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a8a;'>Understanding Air Quality Index and its health impacts</p>", unsafe_allow_html=True)

    guide = [
        ("0–50",   "Good",                           "#00e676",
         "Air quality is satisfactory. No health risk.",
         "✅ Perfect for all outdoor activities including jogging, cycling, and sports."),
        ("51–100", "Moderate",                        "#ffea00",
         "Acceptable for most people. Unusually sensitive individuals may experience minor symptoms.",
         "🟡 Reduce prolonged outdoor exertion if you're unusually sensitive."),
        ("101–150","Unhealthy for Sensitive Groups",  "#ff9100",
         "Children, elderly, and those with respiratory conditions may experience symptoms.",
         "⚠️ Sensitive groups should limit outdoor time. Wear N95 if necessary."),
        ("151–200","Unhealthy",                       "#ff1744",
         "Everyone may begin to experience health effects. Sensitive groups at greater risk.",
         "🔴 Avoid prolonged outdoor exertion. Close windows. Use air purifiers indoors."),
        ("201–300","Very Unhealthy",                  "#d500f9",
         "Health alert: everyone may experience more serious health effects.",
         "🟣 Stay indoors. Wear N95 mask outdoors. Avoid all strenuous activity."),
        ("300+",   "Hazardous",                       "#ff006e",
         "Health emergency. Everyone is likely to be affected seriously.",
         "🚨 DO NOT go outside. Seal windows and doors. Use air purifiers at max setting."),
    ]

    for rng, cat, color, desc, action in guide:
        st.markdown(f"""
        <div style='background:#0d1421; border-left:4px solid {color};
                    border-radius:0 12px 12px 0; padding:20px; margin-bottom:12px;'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>
                <span style='font-family:Space Mono; font-size:1.1rem; color:{color}; font-weight:700;'>AQI {rng}</span>
                <span style='background:{color}22; color:{color}; padding:3px 10px;
                             border-radius:20px; font-size:0.8rem; font-weight:600;'>{cat}</span>
            </div>
            <p style='color:#e8f4f8; margin:4px 0;'>{desc}</p>
            <p style='color:#5a7a8a; font-size:0.85rem; margin:4px 0;'>{action}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Who's Most at Risk ──
    st.markdown("<br><div class='section-header'>Who Is Most at Risk?</div>", unsafe_allow_html=True)
    risk_groups = [
        ("👶", "Children",         "Developing lungs are more susceptible to air pollution damage."),
        ("👴", "Elderly",          "Weakened immune systems and pre-existing conditions increase risk."),
        ("🫁", "Asthma Patients",  "Poor air quality triggers asthma attacks and worsens symptoms."),
        ("🤰", "Pregnant Women",   "Pollution can affect fetal development and birth outcomes."),
        ("🏃", "Outdoor Athletes", "Higher breathing rates mean more pollutant intake during exercise."),
    ]
    cols = st.columns(5)
    for col, (icon, group, desc) in zip(cols, risk_groups):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-weight:600; margin:8px 0;'>{group}</div>
                <div style='color:#5a7a8a; font-size:0.8rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center; color:#5a7a8a; font-size:0.75rem; padding:12px;'>
    🌫️ Karachi AQI Watch &nbsp;|&nbsp;
    Data: OpenWeather API &nbsp;|&nbsp;
    Model: {model_name} (R²={metrics.get('r2', 0):.4f}) &nbsp;|&nbsp;
    Updated: {forecast.get('generated_at', 'N/A')}
</div>
""", unsafe_allow_html=True)