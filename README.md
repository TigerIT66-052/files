# 🏆 Buriram Tourism Dashboard

ระบบ Dashboard วิเคราะห์และทำนายสถิตินักท่องเที่ยวจังหวัดบุรีรัมย์  
ข้อมูลครอบคลุมปี 2556–2568 (พ.ศ.) พร้อมทำนายปี 2569

---

## 🚀 วิธี Deploy บน Streamlit Cloud (ผ่าน GitHub)

### 1. สร้าง GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit: Buriram Tourism Dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 2. วางไฟล์ในโปรเจกต์
```
your-repo/
├── app.py                          ← โค้ดหลัก Streamlit
├── dataCI02-09-03-2569.csv         ← ไฟล์ข้อมูล (ต้องอัพโหลดด้วย!)
├── requirements.txt                ← dependencies
└── README.md
```

> ⚠️ **สำคัญ:** ต้องอัพโหลดไฟล์ `dataCI02-09-03-2569.csv` ขึ้น GitHub ด้วย  
> เพราะ app.py อ่านไฟล์จาก path เดียวกัน: `pd.read_csv("dataCI02-09-03-2569.csv")`

### 3. Deploy บน Streamlit Cloud
1. ไปที่ [share.streamlit.io](https://share.streamlit.io)
2. เชื่อม GitHub account
3. เลือก Repository, Branch: `main`, Main file: `app.py`
4. กด **Deploy** ✅

---

## 🔍 Data Cleaning (10 ขั้นตอน)

| # | การทำความสะอาด |
|---|---|
| 1 | แยกแถวรายเดือน vs รายไตรมาส |
| 2 | ลบแถวซ้ำ/แถวว่าง (quarterly) |
| 3 | ลบแถวซ้ำ/แถวว่าง (monthly) |
| 4 | แปลง String → Numeric (ลบ comma จากตัวเลข) |
| 5 | แปลงปีพุทธศักราช → คริสต์ศักราช (Year - 543) |
| 6 | Map ชื่อไตรมาสภาษาไทย → ตัวเลข 1-4 |
| 7 | Fill NaN ใน Event columns → 0 |
| 8 | แปลง Rev_total → float |
| 9 | นับจำนวนแมตช์ฟุตบอลต่อช่วงเวลา |
| 10 | Sort ตามปี + ไตรมาส |

---

## 🤖 โมเดลที่เปรียบเทียบ

- Linear Regression
- Ridge Regression  
- **Gradient Boosting Regression** ← โมเดลที่เลือกใช้ (R² สูงสุด)
- Random Forest Regression

**Features:** Year_CE, Quarter, MotoGP, Covid, Marathon, PhanomRung_Festival, Football_count  
**Metrics:** MAE, RMSE, R²

---

## 📊 ฟีเจอร์ของ Dashboard

| หน้า | คำอธิบาย |
|------|-----------|
| 🏠 ภาพรวม | KPI, trend รายปี, heatmap, event impact |
| 🔍 Data Cleaning | log ขั้นตอนทำความสะอาด + ตัวอย่างข้อมูล |
| 🤖 Model Comparison | MAE/RMSE/R² ของ 4 โมเดล + actual vs predicted |
| 📈 ทำนายปี 2569 | ทำนายรายไตรมาส ไทย/ต่างชาติ |
| 🎪 อีเวนต์ & ผลกระทบ | impact analysis + ตาราง football matches |
| 📅 สถิติรายปี | ดูข้อมูลย้อนหลังแต่ละปี |
| 📊 Top 5 เดือน | รายงานเดือนที่คาดว่าจะมีนักท่องเที่ยวมากสุด |

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

---

## 💻 Run Locally (VS Code)

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

แล้วเปิดเบราว์เซอร์ไปที่ `http://localhost:8501`
