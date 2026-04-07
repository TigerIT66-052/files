import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard นักท่องเที่ยวบุรีรัมย์",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1e3a5f;
        border-left: 5px solid #2d6a9f; padding-left: 12px; margin: 20px 0 10px 0;
    }
    .event-card {
        background: #f8f9fa; border-radius: 10px; padding: 14px;
        border-left: 4px solid #2d6a9f; margin-bottom: 10px;
    }
    .event-card.negative { border-left-color: #e74c3c; }
    .event-card.positive { border-left-color: #27ae60; }
    .report-box {
        background: linear-gradient(135deg, #f0f7ff, #e8f4e8);
        border-radius: 14px; padding: 20px;
        border: 1px solid #b3d4f0; margin-top: 16px;
    }
    .rank-badge {
        display: inline-block; background: #2d6a9f; color: white;
        border-radius: 50%; width: 28px; height: 28px;
        text-align: center; line-height: 28px; font-weight: bold;
        margin-right: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4f8; border-radius: 8px 8px 0 0; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #2d6a9f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
MONTH_MAP = {
    "มกราคม": 1, "กุมภาพันธ์": 2, "มีนาคม": 3,
    "เมษายน": 4, "พฤษภาคม": 5, "มิถุนายน": 6,
    "กรกฎาคม": 7, "สิงหาคม": 8, "กันยายน": 9,
    "ตุลาคม": 10, "พฤศจิกายน": 11, "ธันวาคม": 12,
    "มกราคม - มีนาคม": 1, "เมษายน - มิถุนายน": 4,
    "กรกฎาคม - กันยายน": 7, "ตุลาคม - ธันวาคม": 10,
}
QUARTER_LABEL = {1: "Q1 (ม.ค.–มี.ค.)", 4: "Q2 (เม.ย.–มิ.ย.)",
                  7: "Q3 (ก.ค.–ก.ย.)", 10: "Q4 (ต.ค.–ธ.ค.)"}
MONTH_TH = {1:"มกราคม", 2:"กุมภาพันธ์", 3:"มีนาคม", 4:"เมษายน",
            5:"พฤษภาคม", 6:"มิถุนายน", 7:"กรกฎาคม", 8:"สิงหาคม",
            9:"กันยายน", 10:"ตุลาคม", 11:"พฤศจิกายน", 12:"ธันวาคม"}
MONTH_SHORT = {1:"ม.ค.", 2:"ก.พ.", 3:"มี.ค.", 4:"เม.ย.", 5:"พ.ค.", 6:"มิ.ย.",
               7:"ก.ค.", 8:"ส.ค.", 9:"ก.ย.", 10:"ต.ค.", 11:"พ.ย.", 12:"ธ.ค."}

# เหตุการณ์และเดือนที่มักเกิด (ข้อมูลเพิ่มเติมสำหรับ Feature 4)
EVENT_MONTHS = {
    "MotoGP":            {"months": [10, 11], "quarter": "Q4 (ต.ค.–พ.ย.)", "label": "🏍️ MotoGP", "color": "#e74c3c"},
    "Marathon":          {"months": [1, 2],   "quarter": "Q1 (ม.ค.–ก.พ.)", "label": "🏃 Marathon", "color": "#3498db"},
    "PhanomRung_Festival":{"months": [4, 5],  "quarter": "Q2 (เม.ย.–พ.ค.)", "label": "🏯 Phanom Rung Festival", "color": "#e67e22"},
    "Covid":             {"months": [3,4,5,6,7,8,9,10,11,12], "quarter": "ตลอดปี (ผลกระทบ 2563–2565)", "label": "🦠 COVID-19", "color": "#8e44ad"},
}

# ─────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(path="dataCI02-09-03-2569.csv"):
    df_raw = pd.read_csv(path)
    df = df_raw[df_raw["Total_vis"].notna()].copy()
    num_cols = ["Total_vis", "Thai_vis", "Foreign_vis", "Guests_total",
                "Rev_total", "Rev_thai", "Rev_foreign"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["month_num"] = df["Month&Quarter"].map(MONTH_MAP)
    quarterly_labels = {"มกราคม - มีนาคม", "เมษายน - มิถุนายน",
                        "กรกฎาคม - กันยายน", "ตุลาคม - ธันวาคม"}
    df["is_quarterly"] = df["Month&Quarter"].isin(quarterly_labels)
    event_cols = ["MotoGP", "Covid", "Marathon", "PhanomRung_Festival"]
    df[event_cols] = df[event_cols].fillna(0)
    df["Year_CE"] = df["Year"] - 543
    df = df.sort_values(["Year", "month_num"]).reset_index(drop=True)
    return df

@st.cache_data
def build_annual(_df):
    annual = (
        _df.groupby("Year")
        .agg(
            Total_vis=("Total_vis", "sum"),
            Thai_vis=("Thai_vis", "sum"),
            Foreign_vis=("Foreign_vis", "sum"),
            Rev_total=("Rev_total", "sum"),
            MotoGP=("MotoGP", "max"),
            Covid=("Covid", "max"),
            Marathon=("Marathon", "max"),
            PhanomRung_Festival=("PhanomRung_Festival", "max"),
        ).reset_index()
    )
    annual["Year_CE"] = annual["Year"] - 543
    annual["prev_vis"] = annual["Total_vis"].shift(1)
    annual["prev2_vis"] = annual["Total_vis"].shift(2)
    annual = annual.dropna(subset=["prev_vis", "prev2_vis"])
    return annual

@st.cache_data
def train_models(_annual):
    feat_cols = ["Year_CE", "prev_vis", "prev2_vis",
                 "MotoGP", "Covid", "Marathon", "PhanomRung_Festival"]
    X = _annual[feat_cols].values
    y = _annual["Total_vis"].values
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        results[name] = {
            "model": model,
            "MAE":  mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "R2":   r2_score(y, y_pred),
            "y_pred": y_pred,
        }
    return results, feat_cols

# ─────────────────────────────────────────────────────────
# PREDICT 2569 (ปี CE = 2026)
# ─────────────────────────────────────────────────────────
def predict_2569(model, annual, event_override=None):
    last   = annual.iloc[-1]
    prev2  = annual.iloc[-2]["Total_vis"] if len(annual) >= 2 else last["Total_vis"]
    events = event_override or {"MotoGP": 1, "Covid": 0, "Marathon": 1, "PhanomRung_Festival": 1}
    X_new  = np.array([[2026, last["Total_vis"], prev2,
                         events["MotoGP"], events["Covid"],
                         events["Marathon"], events["PhanomRung_Festival"]]])
    return float(model.predict(X_new)[0])

# ─────────────────────────────────────────────────────────
# MONTHLY FORECAST 2569
# ─────────────────────────────────────────────────────────
def forecast_monthly_2569(annual, monthly_df, total_2569):
    """
    แจกแจงยอดรวมปี 2569 ตามสัดส่วนเฉลี่ยรายเดือนจากข้อมูลจริง
    """
    if monthly_df.empty or monthly_df["month_num"].isna().all():
        # fallback: กระจายเท่ากัน
        months = list(range(1, 13))
        share  = [1/12]*12
    else:
        pivot = (monthly_df.groupby("month_num")
                 .agg(Total=("Total_vis","mean"),
                      Thai=("Thai_vis","mean"),
                      Foreign=("Foreign_vis","mean"))
                 .reset_index())
        pivot["share"]         = pivot["Total"] / pivot["Total"].sum()
        pivot["thai_share"]    = pivot["Thai"]   / pivot["Thai"].sum()
        pivot["foreign_share"] = pivot["Foreign"]/ pivot["Foreign"].sum()
        return pivot

    return None

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
df     = load_and_clean()
annual = build_annual(df)
model_results, feat_cols = train_models(annual)
best_name  = max(model_results, key=lambda k: model_results[k]["R2"])
best_model = model_results[best_name]["model"]

monthly_df  = df[~df["is_quarterly"]].copy()
quarterly_df = df[df["is_quarterly"]].copy()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Seal_Buriram.svg/200px-Seal_Buriram.svg.png", width=80)
    st.markdown("## 🏍️ Dashboard บุรีรัมย์")
    st.markdown("---")

    selected_tab = st.radio(
        "เมนูหลัก",
        ["🏠 ภาพรวม", "📈 การทำนายปี 2569", "📅 สถิติรายปี",
         "🎪 ผลกระทบเหตุการณ์", "📋 รายงานทำนายรายเดือน", "🤖 เปรียบเทียบโมเดล"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ ตั้งค่าการทำนายปี 2569")
    motogp_2569    = st.selectbox("MotoGP 2569",    [1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    marathon_2569  = st.selectbox("Marathon 2569",  [1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    phanomrung_2569= st.selectbox("Phanom Rung 2569",[1, 0], format_func=lambda x: "มี" if x else "ไม่มี")
    covid_2569     = st.selectbox("COVID 2569",     [0, 1], format_func=lambda x: "มี" if x else "ไม่มี")

    event_settings = {
        "MotoGP": motogp_2569, "Covid": covid_2569,
        "Marathon": marathon_2569, "PhanomRung_Festival": phanomrung_2569
    }
    pred_2569_custom = predict_2569(best_model, annual, event_settings)
    st.markdown(f"**📌 ผลทำนายรวมปี 2569**")
    st.success(f"**{pred_2569_custom:,.0f} คน**")
    st.caption(f"โมเดลที่ใช้: {best_name}")
    st.markdown("---")
    st.caption("ข้อมูล: สถิตินักท่องเที่ยวบุรีรัมย์ 2556–2568")

# ─────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg,#1e3a5f,#2d6a9f);
            padding:24px 32px;border-radius:16px;color:white;margin-bottom:24px;'>
    <h1 style='margin:0;font-size:2rem;'>🏍️ Dashboard นักท่องเที่ยวจังหวัดบุรีรัมย์</h1>
    <p style='margin:6px 0 0;opacity:.85;font-size:1rem;'>
        ข้อมูลสถิติและการพยากรณ์นักท่องเที่ยว ปี 2556–2569
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 1: ภาพรวม
# ═══════════════════════════════════════════════════════════
if selected_tab == "🏠 ภาพรวม":
    latest_year = annual.iloc[-1]
    prev_year   = annual.iloc[-2]

    col1, col2, col3, col4 = st.columns(4)
    def kpi(col, label, value, delta=None):
        delta_html = ""
        if delta is not None:
            color = "lightgreen" if delta >= 0 else "#ff6b6b"
            arrow = "▲" if delta >= 0 else "▼"
            delta_html = f"<div style='color:{color};font-size:.85rem;'>{arrow} {abs(delta):,.0f}</div>"
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{value:,.0f}</div>
            <div class='metric-label'>{label}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    with col1: kpi(col1, f"รวมปี {int(latest_year['Year'])}", latest_year["Total_vis"],
                   delta=latest_year["Total_vis"]-prev_year["Total_vis"])
    with col2: kpi(col2, "นักท่องเที่ยวไทย", latest_year["Thai_vis"])
    with col3: kpi(col3, "นักท่องเที่ยวต่างชาติ", latest_year["Foreign_vis"])
    with col4: kpi(col4, "🔮 คาดการณ์ปี 2569", pred_2569_custom,
                   delta=pred_2569_custom-latest_year["Total_vis"])

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 แนวโน้มนักท่องเที่ยวรายปี (2556–2568)</div>", unsafe_allow_html=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=annual["Year"], y=annual["Total_vis"],
        mode="lines+markers+text", name="จริง",
        line=dict(color="#2d6a9f", width=3), marker=dict(size=8),
        text=[f"{v/1e6:.2f}M" for v in annual["Total_vis"]],
        textposition="top center"
    ))
    fig_trend.add_trace(go.Scatter(
        x=[annual["Year"].iloc[-1], 2569], y=[annual["Total_vis"].iloc[-1], pred_2569_custom],
        mode="lines+markers", name="ทำนาย 2569",
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(size=12, color="#e74c3c")
    ))
    fig_trend.add_annotation(x=2569, y=pred_2569_custom,
        text=f"<b>2569 ≈ {pred_2569_custom/1e6:.2f}M</b>",
        showarrow=True, arrowhead=2, bgcolor="#e74c3c", font=dict(color="white"))
    fig_trend.add_vrect(x0=2562.5, x1=2565.5, fillcolor="rgba(231,76,60,0.1)",
                        line_width=0, annotation_text="COVID-19", annotation_position="top left")
    fig_trend.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                            xaxis_title="ปี (พ.ศ.)", yaxis_title="จำนวนนักท่องเที่ยว (คน)")
    st.plotly_chart(fig_trend, use_container_width=True)

    annual_clean = annual.dropna(subset=["Thai_vis", "Foreign_vis"])
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=annual_clean["Year"], y=annual_clean["Thai_vis"],
                             name="ไทย", marker_color="#2d6a9f"))
    fig_bar.add_trace(go.Bar(x=annual_clean["Year"], y=annual_clean["Foreign_vis"],
                             name="ต่างชาติ", marker_color="#e67e22"))
    fig_bar.update_layout(barmode="stack", height=360, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="ปี (พ.ศ.)", yaxis_title="จำนวน (คน)",
                          title="สัดส่วนนักท่องเที่ยวไทย vs ต่างชาติ")
    st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2: การทำนายปี 2569  (Feature 1)
# ═══════════════════════════════════════════════════════════
elif selected_tab == "📈 การทำนายปี 2569":
    st.markdown("<div class='section-header'>🔮 การพยากรณ์ปี 2569 ด้วยโมเดล ML</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.info(f"""
        **โมเดลที่ดีที่สุด**: {best_name}  
        **R²**: {model_results[best_name]['R2']:.4f}  
        **MAE**: {model_results[best_name]['MAE']:,.0f}  
        **RMSE**: {model_results[best_name]['RMSE']:,.0f}  

        **📌 ผลการทำนายปี 2569**  
        (ตามค่าที่ตั้งใน Sidebar)
        """)
        st.metric("คาดการณ์รวม", f"{pred_2569_custom:,.0f} คน",
                  delta=f"{pred_2569_custom - annual['Total_vis'].iloc[-1]:+,.0f} จากปี 2568")

    with col_b:
        model_names = list(model_results.keys())
        r2_vals = [model_results[m]["R2"] for m in model_names]
        pred_vals = [predict_2569(model_results[m]["model"], annual, event_settings) for m in model_names]

        fig_r2 = go.Figure(go.Bar(
            x=r2_vals, y=model_names, orientation="h",
            marker_color=["#e74c3c" if m == best_name else "#2d6a9f" for m in model_names],
            text=[f"R²={v:.4f}" for v in r2_vals], textposition="inside"
        ))
        fig_r2.update_layout(title="R² Score ของแต่ละโมเดล", height=260,
                             xaxis_range=[0,1], plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")

    # ── การทำนายรายโมเดลทุกโมเดล ──
    st.markdown("<div class='section-header'>📊 ผลทำนายปี 2569 จากทุกโมเดล</div>", unsafe_allow_html=True)
    col_cards = st.columns(4)
    for i, (mname, res) in enumerate(model_results.items()):
        p = predict_2569(res["model"], annual, event_settings)
        with col_cards[i]:
            badge = "🏆 " if mname == best_name else ""
            st.markdown(f"""
            <div class='metric-card' style='{"background:linear-gradient(135deg,#c0392b,#e74c3c);" if mname==best_name else ""}'>
                <div style='font-size:.8rem;opacity:.8;'>{badge}{mname}</div>
                <div class='metric-value' style='font-size:1.4rem;'>{p:,.0f}</div>
                <div class='metric-label'>คน (ปี 2569)</div>
                <div style='font-size:.75rem;opacity:.7;'>R²={res["R2"]:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>📉 ข้อมูลจริง vs ทำนาย</div>", unsafe_allow_html=True)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=annual["Year"], y=annual["Total_vis"],
        mode="lines+markers", name="ข้อมูลจริง",
        line=dict(color="#2d6a9f", width=3)))
    fig_pred.add_trace(go.Scatter(x=annual["Year"],
        y=model_results[best_name]["y_pred"],
        mode="lines+markers", name=f"ทำนาย ({best_name})",
        line=dict(color="#e67e22", width=2, dash="dot")))
    fig_pred.add_trace(go.Scatter(x=[2569], y=[pred_2569_custom],
        mode="markers", name="พยากรณ์ 2569",
        marker=dict(size=16, color="#e74c3c", symbol="star")))
    fig_pred.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="ปี (พ.ศ.)", yaxis_title="นักท่องเที่ยว (คน)")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Sensitivity
    st.markdown("<div class='section-header'>🔧 วิเคราะห์ความไว — Sensitivity Analysis</div>", unsafe_allow_html=True)
    scenarios = []
    for motogp in [0,1]:
        for covid in [0,1]:
            for marathon in [0,1]:
                for phanomrung in [0,1]:
                    ev = {"MotoGP":motogp,"Covid":covid,"Marathon":marathon,"PhanomRung_Festival":phanomrung}
                    p  = predict_2569(best_model, annual, ev)
                    label = (f"MotoGP={'✓' if motogp else '✗'} | COVID={'✓' if covid else '✗'} | "
                             f"Marathon={'✓' if marathon else '✗'} | PhanomRung={'✓' if phanomrung else '✗'}")
                    scenarios.append({"Scenario": label, "Prediction": p})
    df_sc = pd.DataFrame(scenarios).sort_values("Prediction", ascending=False)
    fig_sc = px.bar(df_sc.head(10), x="Prediction", y="Scenario",
                    orientation="h", color="Prediction",
                    color_continuous_scale="Blues",
                    title="Top 10 สถานการณ์คาดการณ์ปี 2569")
    fig_sc.update_layout(height=440, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: สถิติรายปี  (Feature 3)
# ═══════════════════════════════════════════════════════════
elif selected_tab == "📅 สถิติรายปี":
    st.markdown("<div class='section-header'>📅 สถิตินักท่องเที่ยวรายปี — ดูข้อมูลเก่าแต่ละปี</div>",
                unsafe_allow_html=True)

    all_years = sorted(df["Year"].unique(), reverse=True)
    sel_year  = st.selectbox("🔍 เลือกปีที่ต้องการดู (พ.ศ.)", all_years)

    year_data = df[df["Year"] == sel_year].copy()
    is_quarterly = year_data["is_quarterly"].all()

    annual_row = annual[annual["Year"] == sel_year]
    if not annual_row.empty:
        ar = annual_row.iloc[0]
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("รวมทั้งปี", f"{ar['Total_vis']:,.0f} คน")
        col2.metric("นักท่องเที่ยวไทย", f"{ar['Thai_vis']:,.0f} คน" if ar['Thai_vis']>0 else "N/A")
        col3.metric("นักท่องเที่ยวต่างชาติ", f"{ar['Foreign_vis']:,.0f} คน" if ar['Foreign_vis']>0 else "N/A")
        events_this_year = []
        for ev, info in EVENT_MONTHS.items():
            if ar[ev] == 1:
                events_this_year.append(info["label"])
        col4.metric("เหตุการณ์สำคัญ", ", ".join(events_this_year) if events_this_year else "ไม่มี")

    st.markdown("---")

    # แสดงกราฟตามประเภทข้อมูล
    if is_quarterly:
        year_data["quarter_label"] = year_data["month_num"].map(QUARTER_LABEL)
        year_data = year_data.sort_values("month_num")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=year_data["quarter_label"], y=year_data["Total_vis"],
                             marker_color="#27ae60", name="รวม",
                             text=year_data["Total_vis"].apply(lambda x: f"{x:,.0f}"),
                             textposition="outside"))
        if "Thai_vis" in year_data.columns and year_data["Thai_vis"].notna().any():
            fig.add_trace(go.Bar(x=year_data["quarter_label"], y=year_data["Thai_vis"],
                                 marker_color="#2d6a9f", name="ไทย"))
            fig.add_trace(go.Bar(x=year_data["quarter_label"], y=year_data["Foreign_vis"],
                                 marker_color="#e67e22", name="ต่างชาติ"))
        fig.update_layout(title=f"นักท่องเที่ยวรายไตรมาส ปี {sel_year}",
                          barmode="group", height=420,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="ไตรมาส", yaxis_title="จำนวน (คน)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            year_data[["quarter_label","Total_vis","Thai_vis","Foreign_vis",
                       "MotoGP","Covid","Marathon","PhanomRung_Festival"]]
            .rename(columns={"quarter_label":"ไตรมาส","Total_vis":"รวม",
                             "Thai_vis":"ไทย","Foreign_vis":"ต่างชาติ"}),
            use_container_width=True, hide_index=True
        )
    else:
        year_data["month_label"] = year_data["month_num"].map(MONTH_TH)
        year_data = year_data.sort_values("month_num")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=year_data["month_label"], y=year_data["Total_vis"],
                             marker_color="#2d6a9f", name="รวม",
                             text=year_data["Total_vis"].apply(lambda x: f"{x:,.0f}"),
                             textposition="outside"))
        if year_data["Thai_vis"].notna().any():
            fig.add_trace(go.Bar(x=year_data["month_label"], y=year_data["Thai_vis"],
                                 marker_color="#3498db", name="ไทย"))
            fig.add_trace(go.Bar(x=year_data["month_label"], y=year_data["Foreign_vis"],
                                 marker_color="#e67e22", name="ต่างชาติ"))
        # Mark events บนกราฟ
        for _, row in year_data.iterrows():
            events_here = []
            if row["MotoGP"]==1:              events_here.append("🏍️")
            if row["Marathon"]==1:            events_here.append("🏃")
            if row["PhanomRung_Festival"]==1: events_here.append("🏯")
            if row["Covid"]==1:               events_here.append("🦠")
            if events_here:
                fig.add_annotation(
                    x=row["month_label"], y=row["Total_vis"],
                    text=" ".join(events_here), showarrow=False,
                    yshift=18, font=dict(size=14))
        fig.update_layout(title=f"นักท่องเที่ยวรายเดือน ปี {sel_year}",
                          barmode="group", height=440,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="เดือน", yaxis_title="จำนวน (คน)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            year_data[["month_label","Total_vis","Thai_vis","Foreign_vis",
                       "MotoGP","Covid","Marathon","PhanomRung_Festival"]]
            .rename(columns={"month_label":"เดือน","Total_vis":"รวม",
                             "Thai_vis":"ไทย","Foreign_vis":"ต่างชาติ"}),
            use_container_width=True, hide_index=True
        )

    # เปรียบเทียบกับปีก่อน
    st.markdown("<div class='section-header'>📊 เปรียบเทียบกับปีก่อน</div>", unsafe_allow_html=True)
    prev_yr = sel_year - 1
    if prev_yr in annual["Year"].values:
        curr_row = annual[annual["Year"]==sel_year].iloc[0]
        prev_row = annual[annual["Year"]==prev_yr].iloc[0]
        diff = curr_row["Total_vis"] - prev_row["Total_vis"]
        pct  = diff / prev_row["Total_vis"] * 100 if prev_row["Total_vis"] > 0 else 0
        c1,c2,c3 = st.columns(3)
        c1.metric(f"ปี {sel_year} รวม", f"{curr_row['Total_vis']:,.0f}")
        c2.metric(f"ปี {prev_yr} รวม",  f"{prev_row['Total_vis']:,.0f}")
        c3.metric("เปลี่ยนแปลง", f"{diff:+,.0f} ({pct:+.1f}%)",
                  delta_color="normal" if diff>=0 else "inverse")

# ═══════════════════════════════════════════════════════════
# TAB 4: ผลกระทบเหตุการณ์  (Feature 2 + Feature 4)
# ═══════════════════════════════════════════════════════════
elif selected_tab == "🎪 ผลกระทบเหตุการณ์":
    # Feature 4: บอกว่าเหตุการณ์เกิดเดือน/ไตรมาสไหน
    st.markdown("<div class='section-header'>📅 เหตุการณ์สำคัญ — เดือน/ไตรมาสที่เกิดขึ้น</div>",
                unsafe_allow_html=True)

    for ev, info in EVENT_MONTHS.items():
        month_names = [MONTH_TH.get(m, "") for m in info["months"]]
        month_str   = " / ".join(month_names)
        st.markdown(f"""
        <div class='event-card {"negative" if ev=="Covid" else "positive"}'>
            <b>{info["label"]}</b><br>
            📅 <b>เดือนที่เกิด:</b> {month_str}<br>
            📊 <b>ไตรมาส:</b> {info["quarter"]}
        </div>""", unsafe_allow_html=True)

    # ข้อมูลจากไฟล์จริง (ถ้ามี Football_Month)
    if "Football_Month" in df.columns:
        football_df = df[df["Football_match"].notna() & (df["Football_match"] != "")].copy()
        if not football_df.empty:
            st.markdown("**⚽ ฟุตบอล Chang Champions Cup**")
            for _, row in football_df[["Year","Football_Month","Football_date","Football_match"]].drop_duplicates().iterrows():
                st.markdown(f"""
                <div class='event-card'>
                    ⚽ <b>{row['Football_match']}</b> — ปี {int(row['Year'])} 
                    เดือน {row.get('Football_Month','')} วันที่ {row.get('Football_date','')}
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Feature 2: บอกว่าเหตุการณ์ส่งผลกระทบมากน้อยแค่ไหน
    st.markdown("<div class='section-header'>📊 ผลกระทบของเหตุการณ์ต่อจำนวนนักท่องเที่ยว</div>",
                unsafe_allow_html=True)

    events_list  = ["MotoGP", "Marathon", "PhanomRung_Festival", "Covid"]
    event_labels = {"MotoGP":"🏍️ MotoGP","Marathon":"🏃 Marathon",
                    "PhanomRung_Festival":"🏯 Phanom Rung Festival","Covid":"🦠 COVID-19"}

    impact_data = []
    for ev in events_list:
        has = annual[annual[ev]==1]["Total_vis"]
        no  = annual[annual[ev]==0]["Total_vis"]
        if len(has) > 0 and len(no) > 0:
            diff  = has.mean() - no.mean()
            pct   = diff / no.mean() * 100
            impact_data.append({
                "เหตุการณ์": event_labels[ev],
                "เฉลี่ยเมื่อมี": has.mean(),
                "เฉลี่ยเมื่อไม่มี": no.mean(),
                "ผลต่าง": diff,
                "ผลกระทบ (%)": pct,
                "ev_key": ev,
            })
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"**{event_labels[ev]}**")
        if len(has) > 0:
            c2.metric("เฉลี่ยเมื่อมี", f"{has.mean():,.0f}")
        if len(no) > 0:
            c3.metric("เฉลี่ยเมื่อไม่มี", f"{no.mean():,.0f}")
        if len(has) > 0 and len(no) > 0:
            diff = has.mean() - no.mean()
            pct  = diff / no.mean() * 100
            c4.metric("ผลกระทบ", f"{diff:+,.0f} ({pct:+.1f}%)",
                      delta_color="normal" if diff>0 else "inverse")
        st.markdown("---")

    # กราฟแท่งแสดงผลกระทบรวม
    if impact_data:
        df_impact = pd.DataFrame(impact_data)
        fig_impact = px.bar(
            df_impact, x="เหตุการณ์", y="ผลกระทบ (%)",
            color="ผลกระทบ (%)", color_continuous_scale="RdYlGn",
            title="ผลกระทบของแต่ละเหตุการณ์ต่อจำนวนนักท่องเที่ยว (%)",
            text=df_impact["ผลกระทบ (%)"].apply(lambda x: f"{x:+.1f}%")
        )
        fig_impact.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                                 yaxis_title="ผลกระทบ (%)", xaxis_title="")
        st.plotly_chart(fig_impact, use_container_width=True)

    # Feature Importance จาก Random Forest
    if "Random Forest" in model_results:
        st.markdown("<div class='section-header'>🔬 ความสำคัญของแต่ละปัจจัย (Random Forest)</div>",
                    unsafe_allow_html=True)
        importances = model_results["Random Forest"]["model"].feature_importances_
        feat_labels = {
            "Year_CE":"ปีปฏิทิน","prev_vis":"ยอดปีก่อน","prev2_vis":"ยอด 2 ปีก่อน",
            "MotoGP":"🏍️ MotoGP","Covid":"🦠 COVID","Marathon":"🏃 Marathon",
            "PhanomRung_Festival":"🏯 Phanom Rung"
        }
        fi_df = pd.DataFrame({
            "ปัจจัย": [feat_labels.get(f,f) for f in feat_cols],
            "ความสำคัญ (%)": importances*100
        }).sort_values("ความสำคัญ (%)", ascending=True)
        fig_fi = px.bar(fi_df, x="ความสำคัญ (%)", y="ปัจจัย", orientation="h",
                        color="ความสำคัญ (%)", color_continuous_scale="Blues",
                        title="ปัจจัยที่ส่งผลต่อการทำนายนักท่องเที่ยวมากที่สุด")
        fig_fi.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fi, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5: รายงานทำนายรายเดือน  (Feature 5)
# ═══════════════════════════════════════════════════════════
elif selected_tab == "📋 รายงานทำนายรายเดือน":
    st.markdown("<div class='section-header'>📋 รายงาน: คาดการณ์เดือนที่มีนักท่องเที่ยวมากที่สุด ปี 2569</div>",
                unsafe_allow_html=True)

    # คำนวณสัดส่วนรายเดือนจากข้อมูลจริง (2567–2568)
    if not monthly_df.empty and monthly_df["month_num"].notna().any():
        share_df = forecast_monthly_2569(annual, monthly_df, pred_2569_custom)

        if share_df is not None:
            # คาดการณ์รายเดือน ปี 2569
            share_df["est_total"]   = (share_df["share"]         * pred_2569_custom).round(0)
            # ประมาณ Thai/Foreign จากสัดส่วน
            total_thai    = annual.iloc[-1]["Thai_vis"]
            total_foreign = annual.iloc[-1]["Foreign_vis"]
            ratio_thai    = total_thai / (total_thai + total_foreign) if (total_thai+total_foreign) > 0 else 0.9
            # ใช้สัดส่วนรายเดือนจริง ถ้ามี
            if "thai_share" in share_df.columns and share_df["thai_share"].notna().any():
                # สเกลให้รวมได้เท่ากับ total
                thai_total_2569    = pred_2569_custom * ratio_thai
                foreign_total_2569 = pred_2569_custom * (1 - ratio_thai)
                share_df["est_thai"]    = (share_df["thai_share"]    * thai_total_2569).round(0)
                share_df["est_foreign"] = (share_df["foreign_share"] * foreign_total_2569).round(0)
            else:
                share_df["est_thai"]    = (share_df["est_total"] * ratio_thai).round(0)
                share_df["est_foreign"] = (share_df["est_total"] * (1-ratio_thai)).round(0)

            share_df["month_name"] = share_df["month_num"].map(MONTH_TH)
            share_df = share_df.sort_values("est_total", ascending=False).reset_index(drop=True)

            # กราฟ Treemap เดือนที่คาดว่ามีนักท่องเที่ยวมากสุด
            fig_tree = px.treemap(
                share_df,
                path=["month_name"], values="est_total",
                color="est_total", color_continuous_scale="Blues",
                title="ขนาดสัดส่วนคาดการณ์นักท่องเที่ยวรายเดือน ปี 2569"
            )
            fig_tree.update_layout(height=400)
            st.plotly_chart(fig_tree, use_container_width=True)

            # ─── รายงาน: Top 5 เดือน ───
            st.markdown("---")
            st.markdown("<div class='section-header'>🥇 Top 5 เดือนที่คาดว่ามีนักท่องเที่ยวมากที่สุด ปี 2569</div>",
                        unsafe_allow_html=True)

            top5_total   = share_df.head(5)
            top5_thai    = share_df.sort_values("est_thai",    ascending=False).head(5)
            top5_foreign = share_df.sort_values("est_foreign", ascending=False).head(5)

            col_t, col_th, col_fo = st.columns(3)

            def render_top5(col, header, df_top, val_col, color):
                with col:
                    st.markdown(f"**{header}**")
                    for i, (_, row) in enumerate(df_top.iterrows()):
                        events_this = []
                        m = int(row["month_num"])
                        for ev, info in EVENT_MONTHS.items():
                            if ev != "Covid" and event_settings.get(ev, 0)==1 and m in info["months"]:
                                events_this.append(info["label"].split()[0])
                        ev_tag = " ".join(events_this)
                        rank_color = ["#FFD700","#C0C0C0","#CD7F32","#4a90d9","#5cb85c"][i]
                        st.markdown(f"""
                        <div style='background:#f8f9fa;border-radius:10px;padding:10px 14px;
                                    margin-bottom:8px;border-left:4px solid {rank_color};'>
                            <span style='font-size:1.3rem;font-weight:bold;color:{rank_color};'>#{i+1}</span>
                            <b style='font-size:1rem;margin-left:8px;'>{row["month_name"]}</b>
                            <span style='float:right;color:{color};font-weight:bold;'>
                                {row[val_col]:,.0f} คน
                            </span><br>
                            <span style='font-size:.75rem;color:#666;margin-left:28px;'>{ev_tag}</span>
                        </div>""", unsafe_allow_html=True)

            render_top5(col_t,  "🧑‍🤝‍🧑 รวมทั้งหมด",    top5_total,   "est_total",   "#2d6a9f")
            render_top5(col_th, "🇹🇭 นักท่องเที่ยวไทย", top5_thai,    "est_thai",    "#27ae60")
            render_top5(col_fo, "🌍 นักท่องเที่ยวต่างชาติ",top5_foreign,"est_foreign", "#e67e22")

            # ─── ตารางสรุปรายเดือน ───
            st.markdown("---")
            st.markdown("<div class='section-header'>📊 ตารางคาดการณ์รายเดือนทั้งปี 2569</div>",
                        unsafe_allow_html=True)
            display_df = share_df[["month_name","est_total","est_thai","est_foreign"]].copy()
            display_df.columns = ["เดือน","คาดการณ์รวม","คาดการณ์ไทย","คาดการณ์ต่างชาติ"]
            display_df = display_df.sort_values("เดือน", key=lambda x:
                x.map({v:k for k,v in MONTH_TH.items()}))

            # เพิ่มคอลัมน์เหตุการณ์
            def get_events_for_month(month_name):
                m_num = {v:k for k,v in MONTH_TH.items()}.get(month_name)
                evs = []
                for ev, info in EVENT_MONTHS.items():
                    if ev != "Covid" and event_settings.get(ev,0)==1 and m_num in info["months"]:
                        evs.append(info["label"].split()[0] + info["label"].split()[1])
                return " ".join(evs) if evs else "-"
            display_df["เหตุการณ์"] = display_df["เดือน"].apply(get_events_for_month)

            # format numbers
            for col in ["คาดการณ์รวม","คาดการณ์ไทย","คาดการณ์ต่างชาติ"]:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x):,}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # กราฟเส้นรายเดือน
            share_plot = share_df.sort_values("month_num")
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=share_plot["month_name"], y=share_plot["est_total"],
                name="รวม", marker_color="#2d6a9f",
                text=share_plot["est_total"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside"
            ))
            fig_monthly.add_trace(go.Scatter(
                x=share_plot["month_name"], y=share_plot["est_thai"],
                mode="lines+markers", name="ไทย",
                line=dict(color="#27ae60", width=2)
            ))
            fig_monthly.add_trace(go.Scatter(
                x=share_plot["month_name"], y=share_plot["est_foreign"],
                mode="lines+markers", name="ต่างชาติ",
                line=dict(color="#e67e22", width=2)
            ))
            # Mark events
            for ev, info in EVENT_MONTHS.items():
                if ev != "Covid" and event_settings.get(ev, 0)==1:
                    for m in info["months"]:
                        m_name = MONTH_TH.get(m)
                        match = share_plot[share_plot["month_num"]==m]
                        if not match.empty:
                            fig_monthly.add_annotation(
                                x=m_name, y=match["est_total"].values[0],
                                text=info["label"].split()[0], showarrow=False,
                                yshift=22, font=dict(size=16))
            fig_monthly.update_layout(
                title="คาดการณ์นักท่องเที่ยวรายเดือน ปี 2569",
                height=450, plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="เดือน", yaxis_title="จำนวน (คน)"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            # ─── กล่องรายงานสรุป ───
            top_month_total   = top5_total.iloc[0]["month_name"]
            top_month_thai    = top5_thai.iloc[0]["month_name"]
            top_month_foreign = top5_foreign.iloc[0]["month_name"]
            st.markdown(f"""
            <div class='report-box'>
                <h3>📋 สรุปรายงานคาดการณ์ปี 2569</h3>
                <p>จากการวิเคราะห์โดย <b>{best_name}</b> (R²={model_results[best_name]['R2']:.3f})</p>
                <ul>
                    <li>📌 <b>ยอดนักท่องเที่ยวรวมที่คาดการณ์:</b> {pred_2569_custom:,.0f} คน</li>
                    <li>🥇 <b>เดือนที่คาดว่ามีนักท่องเที่ยวรวมมากสุด:</b> {top_month_total}</li>
                    <li>🇹🇭 <b>เดือนที่คาดว่ามีนักท่องเที่ยวไทยมากสุด:</b> {top_month_thai}</li>
                    <li>🌍 <b>เดือนที่คาดว่ามีนักท่องเที่ยวต่างชาติมากสุด:</b> {top_month_foreign}</li>
                    <li>📊 <b>สมมติฐาน:</b> MotoGP={'มี' if event_settings['MotoGP'] else 'ไม่มี'} | 
                        Marathon={'มี' if event_settings['Marathon'] else 'ไม่มี'} | 
                        Phanom Rung={'มี' if event_settings['PhanomRung_Festival'] else 'ไม่มี'} | 
                        COVID={'มี' if event_settings['Covid'] else 'ไม่มี'}</li>
                </ul>
                <p style='font-size:.85rem;color:#666;'>
                    * การคาดการณ์รายเดือนใช้สัดส่วนเฉลี่ยจากข้อมูลรายเดือนปี 2567–2568 
                    มาปรับสเกลตามยอดรวมที่โมเดลทำนาย
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("ไม่พบข้อมูลรายเดือนเพียงพอสำหรับสร้างรายงาน")
    else:
        st.warning("ข้อมูลรายเดือนไม่พร้อมใช้งาน (ต้องใช้ข้อมูลปี 2567–2568)")

# ═══════════════════════════════════════════════════════════
# TAB 6: เปรียบเทียบโมเดล
# ═══════════════════════════════════════════════════════════
elif selected_tab == "🤖 เปรียบเทียบโมเดล":
    st.markdown("<div class='section-header'>🤖 เปรียบเทียบประสิทธิภาพโมเดล ML</div>", unsafe_allow_html=True)

    metrics_data = []
    for name, res in model_results.items():
        metrics_data.append({
            "โมเดล": name,
            "MAE": f"{res['MAE']:,.0f}",
            "RMSE": f"{res['RMSE']:,.0f}",
            "R²": f"{res['R2']:.4f}",
            "ทำนาย 2569": f"{predict_2569(res['model'], annual, event_settings):,.0f}",
            "🏆": "✅ ดีที่สุด" if name == best_name else ""
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    st.markdown("---")

    # Radar
    fig_radar = go.Figure()
    for name, res in model_results.items():
        r2_n  = max(0, res["R2"])
        mae_n = 1 - min(1, res["MAE"] / annual["Total_vis"].mean())
        rmse_n= 1 - min(1, res["RMSE"] / annual["Total_vis"].mean())
        fig_radar.add_trace(go.Scatterpolar(
            r=[r2_n, mae_n, rmse_n, r2_n],
            theta=["R²","1-MAE (norm)","1-RMSE (norm)","R²"],
            fill="toself", name=name
        ))
    fig_radar.update_layout(title="Radar Chart เปรียบเทียบโมเดล",
                            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                            height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Residuals
    st.markdown("<div class='section-header'>📉 Residual Plot</div>", unsafe_allow_html=True)
    sel_model_name = st.selectbox("เลือกโมเดล", list(model_results.keys()))
    residuals = annual["Total_vis"].values - model_results[sel_model_name]["y_pred"]

    fig_res = make_subplots(rows=1, cols=2,
                            subplot_titles=("Residuals over Years","Distribution of Residuals"))
    fig_res.add_trace(go.Scatter(x=annual["Year"], y=residuals,
                                 mode="lines+markers", name="Residuals",
                                 line=dict(color="#e74c3c")), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_res.add_trace(go.Histogram(x=residuals, nbinsx=10,
                                   marker_color="#2d6a9f", name="Distribution"), row=1, col=2)
    fig_res.update_layout(height=380, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_res, use_container_width=True)

    if sel_model_name in ["Random Forest","Gradient Boosting"]:
        st.markdown("<div class='section-header'>📊 Feature Importance</div>", unsafe_allow_html=True)
        importances = model_results[sel_model_name]["model"].feature_importances_
        feat_labels = {
            "Year_CE":"ปีปฏิทิน","prev_vis":"ยอดปีก่อน","prev2_vis":"ยอด 2 ปีก่อน",
            "MotoGP":"🏍️ MotoGP","Covid":"🦠 COVID","Marathon":"🏃 Marathon",
            "PhanomRung_Festival":"🏯 Phanom Rung"
        }
        fi_df = pd.DataFrame({
            "Feature": [feat_labels.get(f,f) for f in feat_cols],
            "Importance": importances
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        title=f"Feature Importance — {sel_model_name}")
        fig_fi.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_fi, use_container_width=True)