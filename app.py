import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Buriram Tourism Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #7c3aed;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #a78bfa; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; }
    .section-header {
        background: linear-gradient(90deg, #7c3aed22, transparent);
        padding: 10px 16px;
        border-left: 3px solid #7c3aed;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e2130;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #9ca3af;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
        color: white !important;
    }
    .prediction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .event-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin: 2px;
    }
    .clean-step {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    """Load CSV and apply all cleaning steps. Returns cleaned df + log."""
    df_raw = pd.read_csv("dataCI02-09-03-2569.csv")
    log = []

    # STEP 1: Separate monthly vs quarterly rows
    monthly_labels = ['มกราคม','กุมภาพันธ์','มีนาคม','เมษายน','พฤษภาคม','มิถุนายน',
                      'กรกฎาคม','สิงหาคม','กันยายน','ตุลาคม','พฤศจิกายน','ธันวาคม']
    quarter_labels = ['มกราคม - มีนาคม','เมษายน - มิถุนายน','กรกฎาคม - กันยายน','ตุลาคม - ธันวาคม']

    df_quarterly = df_raw[df_raw['Month&Quarter'].isin(quarter_labels)].copy()
    df_monthly_raw = df_raw[df_raw['Month&Quarter'].isin(monthly_labels)].copy()
    log.append("✅ **แยกแถวรายเดือน vs รายไตรมาส** — ข้อมูลในไฟล์มีทั้งแถวรายเดือนและรายไตรมาสปนกัน จึงแยกออกก่อนประมวลผล")

    # STEP 2: Keep only rows that have Total_vis (remove duplicate football sub-rows)
    before = len(df_quarterly)
    df_quarterly = df_quarterly[df_quarterly['Total_vis'].notna()].copy()
    removed = before - len(df_quarterly)
    log.append(f"✅ **ลบแถวซ้ำ/แถวว่าง (quarterly)** — ลบ {removed} แถวที่ไม่มีข้อมูลนักท่องเที่ยว (แถว football sub-rows ที่ duplicate)")

    before_m = len(df_monthly_raw)
    df_monthly_clean = df_monthly_raw[df_monthly_raw['Total_vis'].notna()].copy()
    removed_m = before_m - len(df_monthly_clean)
    log.append(f"✅ **ลบแถวซ้ำ/แถวว่าง (monthly)** — ลบ {removed_m} แถวที่ไม่มีข้อมูลนักท่องเที่ยว")

    # STEP 3: Convert string numbers with commas to int/float
    num_cols = ['Total_vis','Thai_vis','Foreign_vis','Guests_total']
    for col in num_cols:
        for d in [df_quarterly, df_monthly_clean]:
            if col in d.columns:
                d[col] = d[col].astype(str).str.replace(',', '').str.strip()
                d[col] = pd.to_numeric(d[col], errors='coerce')
    log.append("✅ **แปลง String → Numeric** — คอลัมน์ Total_vis, Thai_vis, Foreign_vis, Guests_total มี comma (เช่น '325,805') → แปลงเป็น int")

    # STEP 4: Convert Thai Buddhist Year to Christian Year
    for d in [df_quarterly, df_monthly_clean]:
        d['Year_CE'] = d['Year'] - 543
    log.append("✅ **แปลงปีพุทธศักราช → คริสต์ศักราช** — Year (พ.ศ.) - 543 = Year_CE (ค.ศ.) เพื่อใช้ใน time-series model")

    # STEP 5: Map quarters to numeric (1-4)
    quarter_map = {
        'มกราคม - มีนาคม': 1,
        'เมษายน - มิถุนายน': 2,
        'กรกฎาคม - กันยายน': 3,
        'ตุลาคม - ธันวาคม': 4
    }
    df_quarterly['Quarter'] = df_quarterly['Month&Quarter'].map(quarter_map)
    log.append("✅ **Map ชื่อไตรมาส → ตัวเลข** — 'มกราคม - มีนาคม' → 1, ... 'ตุลาคม - ธันวาคม' → 4")

    # STEP 6: Fill NaN in event columns with 0
    event_cols = ['MotoGP','Covid','Marathon','PhanomRung_Festival']
    for d in [df_quarterly, df_monthly_clean]:
        for col in event_cols:
            if col in d.columns:
                d[col] = d[col].fillna(0)
    log.append("✅ **Fill NaN ใน Event columns → 0** — MotoGP, Covid, Marathon, PhanomRung_Festival ที่เป็น NaN หมายความว่าไม่มีอีเวนต์ จึงแทนด้วย 0")

    # STEP 7: Handle Revenue column (Rev_total has commas in some rows)
    for d in [df_quarterly, df_monthly_clean]:
        if 'Rev_total' in d.columns:
            d['Rev_total'] = pd.to_numeric(
                d['Rev_total'].astype(str).str.replace(',',''), errors='coerce'
            )
    log.append("✅ **แปลง Rev_total → float** — ข้อมูลรายได้บางแถวมี comma จึงทำความสะอาดเหมือนกับ visitor columns")

    # STEP 8: Add Football_count per month/quarter
    football_by_period = (
        df_raw[df_raw['Football_match'].notna()]
        .groupby(['Year','Month&Quarter'])
        .size()
        .reset_index(name='Football_count')
    )
    df_quarterly = df_quarterly.merge(football_by_period, on=['Year','Month&Quarter'], how='left')
    df_quarterly['Football_count'] = df_quarterly['Football_count'].fillna(0)
    df_monthly_clean = df_monthly_clean.merge(football_by_period, on=['Year','Month&Quarter'], how='left')
    df_monthly_clean['Football_count'] = df_monthly_clean['Football_count'].fillna(0)
    log.append("✅ **นับจำนวนแมตช์ฟุตบอลต่อช่วงเวลา** — รวมจำนวน Football_match ต่อ Year+Month/Quarter เป็น feature 'Football_count'")

    # STEP 9: Drop rows where Total_vis is still NaN after all cleaning
    before_q = len(df_quarterly)
    df_quarterly = df_quarterly.dropna(subset=['Total_vis'])
    log.append(f"✅ **Drop แถวที่ Total_vis ยัง NaN** — ลบออก {before_q - len(df_quarterly)} แถวสุดท้ายที่ไม่มีข้อมูล")

    # STEP 10: Sort by Year + Quarter
    df_quarterly = df_quarterly.sort_values(['Year_CE','Quarter']).reset_index(drop=True)
    df_monthly_clean = df_monthly_clean.sort_values('Year_CE').reset_index(drop=True)
    log.append("✅ **Sort ตามปี + ไตรมาส** — เรียงข้อมูลตามลำดับเวลาเพื่อใช้ใน time-series analysis")

    return df_quarterly, df_monthly_clean, log


@st.cache_data
def train_and_evaluate(df):
    """Train 4 models, return best model + metrics + predictions for 2026."""
    features = ['Year_CE','Quarter','MotoGP','Covid','Marathon','PhanomRung_Festival','Football_count']
    target = 'Total_vis'

    X = df[features].values
    y = df[target].values

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        cv_r2 = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2').mean()
        results[name] = {
            'model': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2': cv_r2,
            'y_pred': y_pred
        }

    # Best model = highest CV R2 (cross-validation generalisation)
    # For small datasets (n=44), CV_R2 may be negative for complex models due to overfitting.
    # We pick the model with the LEAST negative (i.e. highest) CV_R2, which favours Ridge/Linear.
    best_name = max(results, key=lambda k: results[k]['CV_R2'])

    # Predict 2026 (Year_CE=2026)
    # Events for 2026 (2569 BE): MotoGP Q1=1, Marathon Q1=1, PhanomRung Q2=1
    pred_2026 = []
    event_2026 = {1: {'MotoGP':1,'Marathon':1,'PhanomRung_Festival':0,'Football_count':3},
                  2: {'MotoGP':0,'Marathon':0,'PhanomRung_Festival':1,'Football_count':4},
                  3: {'MotoGP':0,'Marathon':0,'PhanomRung_Festival':0,'Football_count':3},
                  4: {'MotoGP':0,'Marathon':0,'PhanomRung_Festival':0,'Football_count':3}}
    best_model = results[best_name]['model']
    for q in [1,2,3,4]:
        ev = event_2026[q]
        x_pred = [[2026, q, ev['MotoGP'], 0, ev['Marathon'], ev['PhanomRung_Festival'], ev['Football_count']]]
        pred_val = best_model.predict(x_pred)[0]
        pred_2026.append({'Quarter': q, 'Predicted_Total': max(0, pred_val)})

    # Also predict Thai and Foreign separately
    for col, colname in [('Thai_vis','Predicted_Thai'), ('Foreign_vis','Predicted_Foreign')]:
        yc = df[col].values
        mc = type(best_model).__new__(type(best_model))
        mc.__dict__.update(type(best_model)(**{k: v for k, v in best_model.get_params().items()}).__dict__)
        mc = type(best_model)(**best_model.get_params())
        mc.fit(X, yc)
        for i, q in enumerate([1,2,3,4]):
            ev = event_2026[q]
            x_pred = [[2026, q, ev['MotoGP'], 0, ev['Marathon'], ev['PhanomRung_Festival'], ev['Football_count']]]
            pred_2026[i][colname] = max(0, mc.predict(x_pred)[0])

    return results, best_name, pd.DataFrame(pred_2026)


# ─────────────────────────────────────────────
# MONTH MAPPING
# ─────────────────────────────────────────────
month_order = ['มกราคม','กุมภาพันธ์','มีนาคม','เมษายน','พฤษภาคม','มิถุนายน',
               'กรกฎาคม','สิงหาคม','กันยายน','ตุลาคม','พฤศจิกายน','ธันวาคม']
month_num = {m: i+1 for i, m in enumerate(month_order)}
quarter_name = {1:'Q1 ม.ค.-มี.ค.', 2:'Q2 เม.ย.-มิ.ย.', 3:'Q3 ก.ค.-ก.ย.', 4:'Q4 ต.ค.-ธ.ค.'}

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_q, df_m, clean_log = load_and_clean_data()
results, best_name, pred_2026 = train_and_evaluate(df_q)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/th/thumb/1/12/BuriramUnited.png/120px-BuriramUnited.png", width=80)
    st.title("🏆 บึรีรัมย์ Tourism")
    st.caption("สถิติการท่องเที่ยวจังหวัดบุรีรัมย์")
    st.divider()
    
    page = st.radio("📂 เมนูหลัก", [
        "🏠 ภาพรวม Dashboard",
        "🔍 Data Cleaning",
        "🤖 Model Comparison",
        "📈 ทำนายปี 2569",
        "🎪 อีเวนต์ & ผลกระทบ",
        "📅 สถิติรายปี",
        "📊 รายงานทำนาย Top 5 เดือน"
    ])
    
    st.divider()
    st.markdown("**ข้อมูล:** ปี 2556–2568 (พ.ศ.)")
    st.markdown(f"**Model ที่ดีที่สุด:** 🥇 {best_name}")
    r2 = results[best_name]['R2']
    st.markdown(f"**R² Score:** `{r2:.4f}`")


# ─────────────────────────────────────────────
# PAGE 1: DASHBOARD OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 ภาพรวม Dashboard":
    st.title("🏆 Dashboard การท่องเที่ยวบุรีรัมย์")
    st.caption("ข้อมูลสถิติและการคาดการณ์นักท่องเที่ยว จังหวัดบุรีรัมย์")
    
    # Latest year KPIs (2567 BE = 2024 CE = complete year)
    latest = df_q[df_q['Year_CE'] == 2024]
    if len(latest) < 4:
        latest = df_q[df_q['Year_CE'] == df_q['Year_CE'].max()]
    total_latest = latest['Total_vis'].sum()
    thai_latest = latest['Thai_vis'].sum()
    foreign_latest = latest['Foreign_vis'].sum()
    
    prev_year = df_q[df_q['Year_CE'] == (latest['Year_CE'].iloc[0] - 1)]
    total_prev = prev_year['Total_vis'].sum() if len(prev_year) > 0 else total_latest
    yoy = ((total_latest - total_prev) / total_prev * 100) if total_prev > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 นักท่องเที่ยวรวม", f"{int(total_latest):,}", f"{yoy:+.1f}% YoY")
    with col2:
        st.metric("🇹🇭 นักท่องเที่ยวไทย", f"{int(thai_latest):,}", f"{thai_latest/total_latest*100:.1f}%")
    with col3:
        st.metric("✈️ นักท่องเที่ยวต่างชาติ", f"{int(foreign_latest):,}", f"{foreign_latest/total_latest*100:.1f}%")
    with col4:
        pred_total_2026 = pred_2026['Predicted_Total'].sum()
        st.metric("🔮 ทำนายปี 2569", f"{int(pred_total_2026):,}", f"{'↑' if pred_total_2026 > total_latest else '↓'} vs ปีก่อน")

    st.divider()
    
    # Line chart all years
    yearly = df_q.groupby('Year_CE').agg({'Total_vis':'sum','Thai_vis':'sum','Foreign_vis':'sum'}).reset_index()
    yearly['Year_BE'] = yearly['Year_CE'] + 543
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly['Year_BE'], y=yearly['Total_vis'],
        mode='lines+markers', name='รวม', line=dict(color='#a78bfa', width=3),
        marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=yearly['Year_BE'], y=yearly['Thai_vis'],
        mode='lines+markers', name='ไทย', line=dict(color='#34d399', width=2)))
    fig.add_trace(go.Scatter(x=yearly['Year_BE'], y=yearly['Foreign_vis'],
        mode='lines+markers', name='ต่างชาติ', line=dict(color='#f59e0b', width=2)))
    
    # Add 2569 prediction
    pred_sum = pred_2026['Predicted_Total'].sum()
    fig.add_trace(go.Scatter(x=[2568, 2569], 
        y=[yearly[yearly['Year_BE']==2568]['Total_vis'].values[0] if 2568 in yearly['Year_BE'].values else pred_sum, pred_sum],
        mode='lines+markers', name='ทำนาย 2569',
        line=dict(color='#f43f5e', width=2, dash='dash'), marker=dict(size=10, symbol='star')))
    
    fig.update_layout(
        title="นักท่องเที่ยวบุรีรัมย์รายปี (2556–2569)", height=380,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), xaxis=dict(gridcolor='#374151'),
        yaxis=dict(gridcolor='#374151'), legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Quarterly breakdown heatmap
    col1, col2 = st.columns(2)
    with col1:
        pivot = df_q.pivot_table(values='Total_vis', index='Year', columns='Quarter', aggfunc='sum')
        pivot.columns = ['Q1','Q2','Q3','Q4']
        fig2 = px.imshow(pivot, text_auto='.3s', aspect='auto',
            color_continuous_scale='Purples', title='Heatmap นักท่องเที่ยวรายไตรมาส')
        fig2.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Event impact summary
        event_impact = {}
        for ev in ['MotoGP','Marathon','PhanomRung_Festival','Covid']:
            with_ev = df_q[df_q[ev] == 1]['Total_vis'].mean()
            without_ev = df_q[df_q[ev] == 0]['Total_vis'].mean()
            event_impact[ev] = with_ev - without_ev
        
        labels = {'MotoGP':'🏍️ MotoGP','Marathon':'🏃 Marathon','PhanomRung_Festival':'🏛️ Phanom Rung','Covid':'😷 COVID-19'}
        colors = ['#a78bfa' if v > 0 else '#f87171' for v in event_impact.values()]
        fig3 = go.Figure(go.Bar(
            x=list(event_impact.values()),
            y=[labels[k] for k in event_impact],
            orientation='h',
            marker_color=colors
        ))
        fig3.update_layout(title='ผลกระทบของอีเวนต์ต่อจำนวนนักท่องเที่ยว (avg diff)',
            height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'), xaxis=dict(gridcolor='#374151'))
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2: DATA CLEANING
# ─────────────────────────────────────────────
elif page == "🔍 Data Cleaning":
    st.title("🔍 ขั้นตอนการ Clean ข้อมูล")
    st.info(f"ข้อมูลต้นฉบับ: **209 แถว × 25 คอลัมน์** → หลัง clean เหลือ **{len(df_q)} แถว quarterly** + **{len(df_m)} แถว monthly**")
    
    st.markdown("### 📋 สิ่งที่ทำในการ Clean ข้อมูล")
    for i, step in enumerate(clean_log, 1):
        st.markdown(f"""<div class='clean-step'><b>Step {i}:</b> {step}</div>""", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 📊 ตัวอย่างข้อมูลหลัง Clean (Quarterly)")
    
    show_cols = ['Year','Year_CE','Month&Quarter','Quarter','Total_vis','Thai_vis','Foreign_vis',
                 'Occ_rate','Rev_total','MotoGP','Covid','Marathon','PhanomRung_Festival','Football_count']
    display_df = df_q[show_cols].copy()
    display_df = display_df.rename(columns={'Month&Quarter': 'ช่วงเวลา', 'Year_CE': 'ปี ค.ศ.',
                                             'Total_vis': 'นักท่องเที่ยวรวม', 'Thai_vis': 'นักท่องเที่ยวไทย',
                                             'Foreign_vis': 'นักท่องเที่ยวต่างชาติ'})
    st.dataframe(display_df, use_container_width=True, height=400)
    
    st.divider()
    st.markdown("### 📊 สถิติเบื้องต้นหลัง Clean")
    st.dataframe(df_q[['Total_vis','Thai_vis','Foreign_vis','Occ_rate','Rev_total']].describe().round(2), 
                 use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3: MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.title("🤖 เปรียบเทียบโมเดล Machine Learning")
    
    st.info("""
    **Features ที่ใช้:** ปี (Year_CE), ไตรมาส (Quarter), MotoGP, Covid, Marathon, PhanomRung_Festival, Football_count  
    **Target:** จำนวนนักท่องเที่ยวรวม (Total_vis) — รายไตรมาส
    """)
    
    # Metrics table
    metrics_data = []
    for name, res in results.items():
        metrics_data.append({
            'โมเดล': name,
            'MAE': f"{res['MAE']:,.0f}",
            'RMSE': f"{res['RMSE']:,.0f}",
            'R²': f"{res['R2']:.4f}",
            'CV R²': f"{res['CV_R2']:.4f}",
            'สถานะ': '🥇 ดีที่สุด' if name == best_name else ''
        })
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
    
    st.markdown(f"### 🏆 โมเดลที่เลือก: **{best_name}**")
    r = results[best_name]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{r['MAE']:,.0f}", help="ค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์")
    c2.metric("RMSE", f"{r['RMSE']:,.0f}", help="ค่า Root Mean Squared Error")
    c3.metric("R²", f"{r['R2']:.4f}", help="ค่า R-squared (1.0 = สมบูรณ์แบบ)")
    
    st.divider()
    
    # Actual vs Predicted chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(df_q))), y=df_q['Total_vis'],
        mode='lines+markers', name='ค่าจริง', line=dict(color='#34d399', width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(df_q))), y=results[best_name]['y_pred'],
        mode='lines+markers', name='ค่าทำนาย', line=dict(color='#f59e0b', width=2, dash='dot')))
    
    tick_labels = [f"Q{row['Quarter']}/{row['Year']}" for _, row in df_q.iterrows()]
    fig.update_layout(
        title=f"Actual vs Predicted — {best_name}",
        xaxis=dict(tickvals=list(range(len(df_q))), ticktext=tick_labels, tickangle=45),
        height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), xaxis_gridcolor='#374151', yaxis_gridcolor='#374151',
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart comparing all models by R2
    names = list(results.keys())
    r2s = [results[n]['R2'] for n in names]
    colors_bar = ['#a78bfa' if n == best_name else '#6b7280' for n in names]
    fig2 = go.Figure(go.Bar(x=names, y=r2s, marker_color=colors_bar, text=[f"{v:.4f}" for v in r2s], textposition='outside'))
    fig2.update_layout(title='R² Score เปรียบเทียบทุกโมเดล', height=320,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), yaxis=dict(range=[0,1.1], gridcolor='#374151'))
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 4: PREDICT 2026 (2569)
# ─────────────────────────────────────────────
elif page == "📈 ทำนายปี 2569":
    st.title("🔮 ทำนายสถิตินักท่องเที่ยวปี 2569 (2026)")
    
    st.markdown("""
    > **หมายเหตุ:** การทำนายใช้ข้อมูลจากปี 2556–2568 (พ.ศ.) และ assumptions ของอีเวนต์ปี 2569:
    > Q1 = MotoGP ✅ + Marathon ✅ | Q2 = Phanom Rung Festival ✅ | Q3-Q4 = งานปกติ
    """)
    
    pred_2026_display = pred_2026.copy()
    pred_2026_display['ไตรมาส'] = pred_2026_display['Quarter'].map(quarter_name)
    
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    colors_q = ['#a78bfa', '#34d399', '#f59e0b', '#f43f5e']
    for i, row in pred_2026_display.iterrows():
        with cols[i]:
            st.metric(
                f"🔮 {row['ไตรมาส']}",
                f"{int(row['Predicted_Total']):,}",
                f"ไทย: {int(row['Predicted_Thai']):,} | ต่างชาติ: {int(row['Predicted_Foreign']):,}"
            )
    
    st.divider()
    
    # Prediction bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='นักท่องเที่ยวไทย',
        x=pred_2026_display['ไตรมาส'],
        y=pred_2026_display['Predicted_Thai'],
        marker_color='#34d399'
    ))
    fig.add_trace(go.Bar(
        name='นักท่องเที่ยวต่างชาติ',
        x=pred_2026_display['ไตรมาส'],
        y=pred_2026_display['Predicted_Foreign'],
        marker_color='#f59e0b'
    ))
    fig.update_layout(barmode='stack', title='ทำนายนักท่องเที่ยวปี 2569 รายไตรมาส',
        height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151',
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical comparison
    st.markdown("### 📊 เปรียบเทียบกับปีก่อนหน้า")
    recent_years = df_q[df_q['Year_CE'] >= 2020].copy()
    yearly_recent = recent_years.groupby(['Year_CE','Year']).agg({'Total_vis':'sum'}).reset_index()
    yearly_recent['ป ี'] = yearly_recent['Year'].astype(str) + ' (จริง)'
    
    pred_row = pd.DataFrame([{
        'Year_CE': 2026, 'Year': 2569,
        'Total_vis': pred_2026['Predicted_Total'].sum(),
        'ปี': '2569 (ทำนาย)'
    }])
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=yearly_recent['Year'].astype(str), y=yearly_recent['Total_vis'],
        name='ค่าจริง', marker_color='#6b7280'))
    fig2.add_trace(go.Bar(x=['2569'], y=[pred_2026['Predicted_Total'].sum()],
        name='ทำนาย 2569', marker_color='#a78bfa'))
    fig2.update_layout(title='เปรียบเทียบจำนวนนักท่องเที่ยวรวมรายปี (ล่าสุด)',
        height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151',
        legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 5: EVENTS & IMPACT
# ─────────────────────────────────────────────
elif page == "🎪 อีเวนต์ & ผลกระทบ":
    st.title("🎪 อีเวนต์สำคัญ & ผลกระทบต่อนักท่องเที่ยว")
    
    st.markdown("### 📅 อีเวนต์เกิดในไตรมาสใด")
    event_quarters = {
        '🏍️ MotoGP': 'Q1 (มกราคม – มีนาคม)',
        '🏃 Marathon': 'Q1 (มกราคม – มีนาคม)',
        '🏛️ Phanom Rung Festival': 'Q2 (เมษายน – มิถุนายน)',
        '⚽ ฟุตบอล (ตลอดฤดูกาล)': 'Q1 – Q4 (ทุกไตรมาส)',
        '😷 COVID-19': 'Q1 2563 – Q4 2565 (ส่งผลลบ)'
    }
    col1, col2 = st.columns(2)
    for i, (ev, q) in enumerate(event_quarters.items()):
        with (col1 if i % 2 == 0 else col2):
            color = '#f87171' if 'COVID' in ev else '#a78bfa'
            st.markdown(f"""
            <div style='background:#1e2130;border-radius:8px;padding:14px 18px;margin:6px 0;border-left:3px solid {color}'>
            <b style='font-size:1.05rem'>{ev}</b><br>
            <span style='color:#9ca3af;font-size:0.9rem'>📅 {q}</span>
            </div>""", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 📊 ผลกระทบเชิงสถิติของแต่ละอีเวนต์")
    
    for ev, label, color in [
        ('MotoGP', '🏍️ MotoGP', '#a78bfa'),
        ('Marathon', '🏃 Marathon', '#34d399'),
        ('PhanomRung_Festival', '🏛️ Phanom Rung', '#f59e0b'),
        ('Covid', '😷 COVID-19', '#f87171')
    ]:
        with_ev = df_q[df_q[ev] == 1]['Total_vis']
        without_ev = df_q[df_q[ev] == 0]['Total_vis']
        avg_with = with_ev.mean() if len(with_ev) > 0 else 0
        avg_without = without_ev.mean() if len(without_ev) > 0 else 0
        diff = avg_with - avg_without
        pct = (diff / avg_without * 100) if avg_without > 0 else 0
        n_with = len(with_ev)

        arrow = "▲" if diff > 0 else "▼"
        diff_color = "#34d399" if diff > 0 else "#f87171"
        
        st.markdown(f"""
        <div style='background:#1e2130;border-radius:10px;padding:16px 20px;margin:8px 0;border-left:3px solid {color}'>
        <b style='font-size:1rem'>{label}</b> — มี {n_with} ไตรมาสที่มีอีเวนต์นี้<br>
        <span style='color:#9ca3af'>เฉลี่ยเมื่อมีอีเวนต์:</span> <b>{avg_with:,.0f}</b> &nbsp;|&nbsp; 
        <span style='color:#9ca3af'>เฉลี่ยเมื่อไม่มี:</span> <b>{avg_without:,.0f}</b><br>
        <span style='color:{diff_color};font-size:1.1rem;font-weight:bold'>{arrow} {abs(diff):,.0f} คน ({pct:+.1f}%)</span>
        </div>""", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚽ ตารางแมตช์ฟุตบอล (จากข้อมูล)")
    df_raw = pd.read_csv("dataCI02-09-03-2569.csv")
    football_df = df_raw[df_raw['Football_match'].notna()][['Year','Month&Quarter','Football_date','Football_match']].copy()
    football_df.columns = ['ปี (พ.ศ.)','เดือน/ไตรมาส','วันที่แข่ง','คู่แข่ง']
    
    year_sel = st.selectbox("เลือกปี", sorted(football_df['ปี (พ.ศ.)'].unique(), reverse=True))
    st.dataframe(football_df[football_df['ปี (พ.ศ.)'] == year_sel], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 6: HISTORICAL STATS
# ─────────────────────────────────────────────
elif page == "📅 สถิติรายปี":
    st.title("📅 สถิติย้อนหลังรายปี")
    
    years_available = sorted(df_q['Year'].unique(), reverse=True)
    selected_year = st.selectbox("เลือกปี (พ.ศ.)", years_available)
    
    year_data = df_q[df_q['Year'] == selected_year].copy()
    year_data['ไตรมาส'] = year_data['Quarter'].map(quarter_name)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 นักท่องเที่ยวรวมทั้งปี", f"{int(year_data['Total_vis'].sum()):,}")
    c2.metric("🇹🇭 นักท่องเที่ยวไทย", f"{int(year_data['Thai_vis'].sum()):,}")
    c3.metric("✈️ นักท่องเที่ยวต่างชาติ", f"{int(year_data['Foreign_vis'].sum()):,}")
    
    # Quarterly bar
    fig = go.Figure()
    fig.add_trace(go.Bar(x=year_data['ไตรมาส'], y=year_data['Thai_vis'], name='ไทย', marker_color='#34d399'))
    fig.add_trace(go.Bar(x=year_data['ไตรมาส'], y=year_data['Foreign_vis'], name='ต่างชาติ', marker_color='#f59e0b'))
    fig.update_layout(
        barmode='stack', title=f'นักท่องเที่ยวปี {selected_year} รายไตรมาส',
        height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151',
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show events in that year
    ev_cols = ['MotoGP','Covid','Marathon','PhanomRung_Festival']
    ev_labels = {'MotoGP':'🏍️ MotoGP','Covid':'😷 COVID-19','Marathon':'🏃 Marathon','PhanomRung_Festival':'🏛️ Phanom Rung'}
    
    st.markdown("### 🎪 อีเวนต์ในปีนี้")
    for _, row in year_data.iterrows():
        events = [ev_labels[e] for e in ev_cols if row[e] == 1]
        if events:
            ev_str = " | ".join(events)
            st.markdown(f"**{row['ไตรมาส']}:** {ev_str}")
        else:
            st.markdown(f"**{row['ไตรมาส']}:** — ไม่มีอีเวนต์พิเศษ")
    
    st.divider()
    st.markdown("### 📋 ข้อมูลรายละเอียด")
    detail = year_data[['ไตรมาส','Total_vis','Thai_vis','Foreign_vis','Occ_rate','Rev_total','Football_count']].copy()
    detail.columns = ['ไตรมาส','นักท่องเที่ยวรวม','ไทย','ต่างชาติ','อัตราเข้าพัก (%)','รายได้ (ล้านบาท)','แมตช์ฟุตบอล']
    st.dataframe(detail, use_container_width=True, hide_index=True)
    
    # All-years trend for reference
    st.divider()
    st.markdown("### 📈 แนวโน้มทุกปีเทียบกัน")
    all_yearly = df_q.groupby('Year').agg({'Total_vis':'sum','Thai_vis':'sum','Foreign_vis':'sum'}).reset_index()
    fig2 = px.bar(all_yearly, x='Year', y='Total_vis', color_discrete_sequence=['#7c3aed'],
                  title='นักท่องเที่ยวรวมรายปี (พ.ศ.)')
    fig2.add_vline(x=selected_year, line_dash='dash', line_color='#f59e0b', 
                   annotation_text=f"ปี {selected_year}", annotation_font_color='#f59e0b')
    fig2.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151')
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 7: TOP 5 MONTHS REPORT
# ─────────────────────────────────────────────
elif page == "📊 รายงานทำนาย Top 5 เดือน":
    st.title("📊 รายงานทำนาย Top 5 เดือนปี 2569")
    st.caption("คาดการณ์เดือนที่มีนักท่องเที่ยวไทย/ต่างชาติมากที่สุด 5 เดือนแรก")
    
    # Use monthly data for seasonal pattern
    if len(df_m) > 0:
        monthly_avg = df_m.groupby('Month&Quarter').agg({
            'Total_vis':'mean', 'Thai_vis':'mean', 'Foreign_vis':'mean'
        }).reset_index()
        monthly_avg['month_num'] = monthly_avg['Month&Quarter'].map(month_num)
        monthly_avg = monthly_avg.dropna(subset=['month_num'])
        monthly_avg = monthly_avg.sort_values('month_num')
        
        # Scale to 2569 predicted annual total
        pred_annual = pred_2026['Predicted_Total'].sum()
        pred_thai_annual = pred_2026['Predicted_Thai'].sum()
        pred_foreign_annual = pred_2026['Predicted_Foreign'].sum()
        
        total_monthly_avg = monthly_avg['Total_vis'].sum()
        thai_monthly_avg = monthly_avg['Thai_vis'].sum()
        foreign_monthly_avg = monthly_avg['Foreign_vis'].sum()
        
        monthly_avg['Pred_2569_Total'] = (monthly_avg['Total_vis'] / total_monthly_avg * pred_annual).astype(int)
        monthly_avg['Pred_2569_Thai'] = (monthly_avg['Thai_vis'] / thai_monthly_avg * pred_thai_annual).astype(int)
        monthly_avg['Pred_2569_Foreign'] = (monthly_avg['Foreign_vis'] / foreign_monthly_avg * pred_foreign_annual).astype(int)
        
        st.markdown("### 🥇 Top 5 เดือนที่มีนักท่องเที่ยวรวมมากที่สุด (2569)")
        top5_total = monthly_avg.nlargest(5, 'Pred_2569_Total')[['Month&Quarter','Pred_2569_Total','Pred_2569_Thai','Pred_2569_Foreign']]
        top5_total.columns = ['เดือน','ทำนายรวม','ทำนายไทย','ทำนายต่างชาติ']
        top5_total['อันดับ'] = ['🥇','🥈','🥉','4️⃣','5️⃣']
        
        for _, row in top5_total.iterrows():
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1e2130,#2d3250);border-radius:10px;padding:16px 20px;margin:8px 0;border-left:4px solid #a78bfa'>
            <span style='font-size:1.4rem'>{row['อันดับ']}</span> &nbsp;
            <b style='font-size:1.1rem;color:#a78bfa'>{row['เดือน']}</b><br>
            <span style='color:#9ca3af'>รวม: </span><b style='color:#e5e7eb'>{int(row['ทำนายรวม']):,}</b> &nbsp;|&nbsp;
            <span style='color:#34d399'>🇹🇭 ไทย: {int(row['ทำนายไทย']):,}</span> &nbsp;|&nbsp;
            <span style='color:#f59e0b'>✈️ ต่างชาติ: {int(row['ทำนายต่างชาติ']):,}</span>
            </div>""", unsafe_allow_html=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🇹🇭 Top 5 เดือน นักท่องเที่ยวไทย")
            top5_thai = monthly_avg.nlargest(5, 'Pred_2569_Thai')[['Month&Quarter','Pred_2569_Thai']]
            fig = px.bar(top5_thai, x='Month&Quarter', y='Pred_2569_Thai',
                color_discrete_sequence=['#34d399'], title='Top 5 เดือน (ไทย)')
            fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151', showlegend=False,
                xaxis_title='', yaxis_title='จำนวนนักท่องเที่ยว')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ✈️ Top 5 เดือน นักท่องเที่ยวต่างชาติ")
            top5_foreign = monthly_avg.nlargest(5, 'Pred_2569_Foreign')[['Month&Quarter','Pred_2569_Foreign']]
            fig2 = px.bar(top5_foreign, x='Month&Quarter', y='Pred_2569_Foreign',
                color_discrete_sequence=['#f59e0b'], title='Top 5 เดือน (ต่างชาติ)')
            fig2.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'), yaxis_gridcolor='#374151', showlegend=False,
                xaxis_title='', yaxis_title='จำนวนนักท่องเที่ยว')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.markdown("### 📈 ทำนายรายเดือนทั้งปี 2569")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=monthly_avg['Month&Quarter'], y=monthly_avg['Pred_2569_Thai'],
            mode='lines+markers', name='🇹🇭 ไทย', line=dict(color='#34d399', width=2), fill='tozeroy',
            fillcolor='rgba(52,211,153,0.1)'))
        fig3.add_trace(go.Scatter(x=monthly_avg['Month&Quarter'], y=monthly_avg['Pred_2569_Foreign'],
            mode='lines+markers', name='✈️ ต่างชาติ', line=dict(color='#f59e0b', width=2)))
        fig3.update_layout(
            title='คาดการณ์นักท่องเที่ยวรายเดือน ปี 2569',
            height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'), xaxis_gridcolor='#374151', yaxis_gridcolor='#374151',
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Full table
        st.markdown("### 📋 ตารางทำนายรายเดือนปี 2569 ครบทุกเดือน")
        full_table = monthly_avg[['Month&Quarter','Pred_2569_Total','Pred_2569_Thai','Pred_2569_Foreign']].copy()
        full_table.columns = ['เดือน','ทำนายรวม','ทำนายไทย','ทำนายต่างชาติ']
        full_table['สัดส่วนไทย (%)'] = (full_table['ทำนายไทย'] / full_table['ทำนายรวม'] * 100).round(1)
        st.dataframe(full_table, use_container_width=True, hide_index=True)
    else:
        st.warning("ไม่มีข้อมูลรายเดือนเพียงพอสำหรับทำนาย กรุณาตรวจสอบข้อมูล")
