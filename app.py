"""
ระบบทำนายการยกเลิกบริการของลูกค้า (Telco Churn Prediction) - SVM
รันด้วย:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ================================================================ config
APP_TITLE = "ระบบทำนายการยกเลิกบริการลูกค้า"
APP_SUBTITLE = "TELCO CHURN PREDICTION · SVM"
DEVELOPERS = ["นายกฤษกร พยอมหอม", "นายณนชกร เปรมทอง"]

BASE_DIR = Path(__file__).parent
SCALER_PATH = BASE_DIR / "telco_churn_scaler.joblib"
MODEL_PATH = BASE_DIR / "telco_churn_svm.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

CLASS_LABELS = {0: "มีแนวโน้มใช้บริการต่อ", 1: "มีแนวโน้มยกเลิกบริการ"}
POSITIVE_CLASS = 1

# ฟีเจอร์ของ Titanic ที่ต้องไม่ปรากฏในโมเดล (ใช้ตรวจว่าไฟล์ถูกเทรนผิดชุดข้อมูล)
TITANIC_FEATURES = {"Pclass", "Sex_female", "Age", "Fare", "FamilySize"}

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="centered")

# ================================================================ style
# >>> คัดลอกบล็อก st.markdown("""<style> ... </style>""") ของเดิมมาวางตรงนี้ ไม่ต้องแก้ <<<

# >>> คัดลอก MASCOT_SVG ของเดิมมาวางตรงนี้ ไม่ต้องแก้ <<<


# ================================================================ loaders
@st.cache_resource
def load_artifacts():
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)


def load_accuracy():
    """คืนค่า accuracy จาก metrics.json ถ้าไม่มีค่าจริงให้คืน None (ไม่มีค่าเริ่มต้นปลอม)"""
    try:
        if METRICS_PATH.exists():
            v = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("accuracy")
            return float(v) if v is not None else None
    except Exception:
        pass
    return None


scaler, model = load_artifacts()
features = list(getattr(scaler, "feature_names_in_", []))

# ตรวจว่าโมเดลไม่ใช่ของ Titanic และมีฟีเจอร์ Telco ครบ
if not features or TITANIC_FEATURES & set(features):
    st.error(
        "ไฟล์โมเดล/scaler ยังเป็นของชุดข้อมูล Titanic หรือไม่มีชื่อฟีเจอร์ "
        "กรุณาเทรนใหม่บน Telco แล้วแทนที่ไฟล์ .joblib ก่อนใช้งาน"
    )
    st.stop()
if not {"tenure", "MonthlyCharges", "SeniorCitizen"} <= set(features):
    st.error(f"ฟีเจอร์ในโมเดลไม่ตรงกับฟอร์ม: {features}")
    st.stop()
if not hasattr(model, "predict_proba"):
    st.error("โมเดลนี้เทรนโดยไม่ใช้ probability=True จึงแสดงความน่าจะเป็นไม่ได้ กรุณาเทรนใหม่")
    st.stop()

# ================================================================ header
st.markdown(
    f'<div class="hero">{MASCOT_SVG}<h1>{APP_TITLE}</h1><div class="sub">{APP_SUBTITLE}</div></div>',
    unsafe_allow_html=True,
)

# ================================================================ form
st.markdown('<div class="section">📝 กรอกข้อมูลลูกค้า</div>', unsafe_allow_html=True)
with st.form("customer_form"):
    c1, c2 = st.columns(2)
    tenure = c1.number_input("อายุการใช้บริการ (tenure: เดือน)", 0, 72, 12)
    monthly = c2.number_input("ค่าบริการรายเดือน (MonthlyCharges)", 18.25, 118.75, 65.0, step=1.0)
    senior = c1.selectbox("ผู้สูงอายุ (SeniorCitizen)", ["ไม่ใช่", "ใช่"]) == "ใช่"
    gender = c2.selectbox("เพศ (gender)", ["Female", "Male"])
    contract = c1.selectbox("ประเภทสัญญา (Contract)", ["Month-to-month", "One year", "Two year"])
    submitted = st.form_submit_button("🔍 ทำนายผล")

# ================================================================ result
if submitted:
    # ตรวจชื่อคอลัมน์ one-hot ให้ตรงกับตอนเทรน
    for key in (f"gender_{gender}", f"Contract_{contract}"):
        if key not in features:
            st.error(f"ไม่พบฟีเจอร์ {key} ในโมเดล กรุณาตรวจชื่อคอลัมน์ตอนเทรน")
            st.stop()

    row = {c: 0.0 for c in features}
    row["tenure"] = float(tenure)
    row["MonthlyCharges"] = float(monthly)
    row["SeniorCitizen"] = 1.0 if senior else 0.0
    row[f"gender_{gender}"] = 1.0
    row[f"Contract_{contract}"] = 1.0

    X = scaler.transform(pd.DataFrame([row], columns=features))
    p = float(model.predict_proba(X)[0][1])  # ความน่าจะเป็น churn จากโมเดลล้วน ๆ
    pred = int(p >= 0.5)

    css, icon = ("warn", "⚠️") if pred == POSITIVE_CLASS else ("ok", "😊")
    st.markdown(
        f'<div class="card {css}"><div class="big">{icon} {CLASS_LABELS[pred]}</div>'
        f'<div style="margin-top:4px">ความน่าจะเป็นที่จะยกเลิกบริการ (จากโมเดล) <b>{p*100:.1f}%</b></div>'
        f'<div class="bar"><div style="width:{p*100:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )

# ================================================================ footer
acc = load_accuracy()
acc_txt = f"{acc*100:.2f}%" if acc is not None else "ยังไม่มีค่า accuracy"
st.markdown(
    f"""
<div class="footer">
<div class="lbl">ACCURACY · ค่าความแม่นยำของระบบ (ชุดทดสอบ)</div>
<div class="acc">{acc_txt}</div>
<div class="dev">พัฒนาโดย {" &nbsp;·&nbsp; ".join(DEVELOPERS)}</div>
</div>
""",
    unsafe_allow_html=True,
)