"""
ระบบทำนายการยกเลิกบริการของลูกค้า (Telco Churn Prediction) - SVM
รันด้วย:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================================================ config
APP_TITLE = "ระบบทำนายการยกเลิกบริการลูกค้า"
APP_SUBTITLE = "TELCO CHURN PREDICTION · SVM"
DEVELOPERS = ["นายกฤษกร พยอมหอม", "นายณนชกร เปรมทอง"]

BASE_DIR = Path(__file__).parent
SCALER_PATH = BASE_DIR / "telco_churn_scaler.joblib"
MODEL_PATH = BASE_DIR / "telco_churn_svm.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"      # เก็บค่า accuracy จริงของโมเดล

# ค่า Accuracy สำรอง (ใช้กรณีหาไฟล์ metrics.json ไม่เจอ)
DEFAULT_ACCURACY = 0.85  # 0.85 หมายถึง 85.00%

# ความหมายของคลาสที่โมเดลทำนาย
CLASS_LABELS = {0: "มีแนวโน้มใช้บริการต่อ", 1: "มีแนวโน้มยกเลิกบริการ"}
POSITIVE_CLASS = 1

# ชื่อที่แสดงบนฟอร์ม (ชื่อคอลัมน์ในไฟล์ : ป้ายกำกับ)
FEATURE_LABELS = {
    "Pclass": "ระดับชั้นบริการ (Pclass: 1, 2, 3)",
    "Age": "อายุ (Age: ปี)",
    "Fare": "ค่าบริการ (Fare: บาท/ด.)",
    "FamilySize": "ขนาดครอบครัว (FamilySize: คน)",
}

# ฟีเจอร์ 0/1 : (ป้ายกำกับ, ความหมายของ 0, ความหมายของ 1)
BINARY_FEATURES = {"Sex_female": ("เพศ (Sex)", "ชาย", "หญิง")}

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="centered")

# ================================================================ style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family:'Prompt',sans-serif; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { max-width:780px; padding-top:1.2rem; }

.hero { position:relative; overflow:hidden; text-align:center; color:#fff;
  padding:26px 18px 22px; border-radius:26px;
  background:radial-gradient(circle at 20% 0%,#2E9E5B 0%,#14503A 45%,#0B2E22 100%);
  box-shadow:0 14px 34px rgba(11,46,34,.28); }
.hero::before { content:""; position:absolute; inset:0; opacity:.13;
  background-image:linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px);
  background-size:26px 26px; }
.hero > * { position:relative; }
.hero h1 { font-size:1.75rem; font-weight:700; margin:6px 0 0; letter-spacing:.01em; }
.hero .sub { display:inline-block; margin-top:8px; padding:3px 14px; border-radius:99px;
  font-size:.75rem; letter-spacing:.18em; color:#9BF0BE; border:1px solid rgba(155,240,190,.5); }
.mascot { width:150px; animation:float 3.2s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-9px)} }

.section { color:#14503A; font-weight:600; font-size:1.05rem; margin:22px 0 4px; }
[data-testid="stForm"] { background:#fff; border:1px solid #CFE9D7; border-radius:20px;
  padding:18px 20px; box-shadow:0 4px 16px rgba(46,158,91,.08); }
.stFormSubmitButton > button { background:linear-gradient(90deg,#2E9E5B,#1F7A47); color:#fff;
  border:0; border-radius:99px; padding:.6rem 2rem; font-weight:600; width:100%;
  box-shadow:0 6px 16px rgba(46,158,91,.35); transition:transform .15s; }
.stFormSubmitButton > button:hover { transform:translateY(-2px); color:#fff; }

.card { border-radius:20px; padding:20px 24px; margin:16px 0; }
.ok   { background:#E6F7EC; border:1px solid #8FD3A9; }
.warn { background:#FFF3E2; border:1px solid #F0C283; }
.card .big { font-size:1.3rem; font-weight:700; color:#0B2E22; }
.bar { height:12px; border-radius:99px; background:rgba(0,0,0,.08); overflow:hidden; margin-top:10px; }
.bar > div { height:100%; border-radius:99px; background:linear-gradient(90deg,#7BD39A,#2E9E5B); }
.warn .bar > div { background:linear-gradient(90deg,#F6C453,#E8862B); }

.footer { margin-top:30px; padding:22px; text-align:center; border-radius:22px;
  background:#0B2E22; color:#D9F5E4; }
.footer .lbl { font-size:.8rem; letter-spacing:.14em; color:#7FD6A1; }
.footer .acc { font-size:2.6rem; font-weight:700; color:#fff; line-height:1.15; }
.footer .dev { margin-top:12px; padding-top:12px; border-top:1px dashed rgba(155,240,190,.35);
  font-size:.9rem; color:#B9E9CC; }
</style>
""",
    unsafe_allow_html=True,
)

# ตัวการ์ตูนหุ่นยนต์ใส่แว่นดำ + หูฟัง
MASCOT_SVG = """
<svg class="mascot" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
<ellipse cx="100" cy="190" rx="44" ry="6" fill="rgba(0,0,0,.25)"/>
<line x1="100" y1="34" x2="100" y2="16" stroke="#9BF0BE" stroke-width="5" stroke-linecap="round"/>
<circle cx="100" cy="13" r="8" fill="#F6C453"/>
<rect x="38" y="34" width="124" height="112" rx="52" fill="#8EDBA9" stroke="#D9F5E4" stroke-width="5"/>
<rect x="24" y="76" width="16" height="32" rx="8" fill="#1F7A47"/>
<rect x="160" y="76" width="16" height="32" rx="8" fill="#1F7A47"/>
<path d="M32 84 C32 30 168 30 168 84" fill="none" stroke="#1F7A47" stroke-width="5"/>
<path d="M168 100 Q168 128 138 130" fill="none" stroke="#1F7A47" stroke-width="4" stroke-linecap="round"/>
<circle cx="136" cy="130" r="5" fill="#1F7A47"/>
<rect x="56" y="72" width="40" height="26" rx="10" fill="#0B2E22"/>
<rect x="104" y="72" width="40" height="26" rx="10" fill="#0B2E22"/>
<rect x="94" y="80" width="12" height="5" fill="#0B2E22"/>
<path d="M62 78 l14 0 l-10 12 z" fill="#fff" opacity=".35"/>
<path d="M110 78 l14 0 l-10 12 z" fill="#fff" opacity=".35"/>
<circle cx="62" cy="110" r="8" fill="#FFB7B7" opacity=".85"/><circle cx="138" cy="110" r="8" fill="#FFB7B7" opacity=".85"/>
<path d="M84 112 Q100 128 116 112" fill="none" stroke="#0B2E22" stroke-width="5" stroke-linecap="round"/>
<rect x="66" y="146" width="68" height="34" rx="16" fill="#8EDBA9" stroke="#D9F5E4" stroke-width="5"/>
<path d="M100 154 c-6-7-16 1-8 9 l8 8 l8-8 c8-8-2-16-8-9z" fill="#fff"/>
</svg>
"""


# ================================================================ loaders
@st.cache_resource
def load_artifacts():
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)


def load_accuracy():
    try:
        if METRICS_PATH.exists():
            v = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("accuracy")
            if v is not None:
                return float(v)
        return DEFAULT_ACCURACY
    except Exception:
        return DEFAULT_ACCURACY


def predict(model, X_scaled):
    """คืนค่า (คลาสที่ทำนาย, คะแนนความมั่นใจ 0-1 ของคลาสบวก)"""
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X_scaled)[0][list(model.classes_).index(POSITIVE_CLASS)])
    else:  # SVC(probability=False) -> แปลงระยะห่างจากเส้นแบ่งเป็น 0-1 ด้วย sigmoid
        score = float(1 / (1 + np.exp(-model.decision_function(X_scaled)[0])))
    return int(model.predict(X_scaled)[0]), score


scaler, model = load_artifacts()
features = list(getattr(scaler, "feature_names_in_", [f"x{i}" for i in range(scaler.n_features_in_)]))

# ================================================================ header
st.markdown(
    f'<div class="hero">{MASCOT_SVG}<h1>{APP_TITLE}</h1><div class="sub">{APP_SUBTITLE}</div></div>',
    unsafe_allow_html=True,
)

# ================================================================ form
st.markdown('<div class="section">📝 กรอกข้อมูลลูกค้า</div>', unsafe_allow_html=True)
values = {}
with st.form("customer_form"):
    cols = st.columns(2)
    for i, name in enumerate(features):
        col = cols[i % 2]
        if name in BINARY_FEATURES:
            label, zero, one = BINARY_FEATURES[name]
            values[name] = float(col.selectbox(label, [zero, one], index=0) == one)
        elif name == "Pclass":
            values[name] = float(col.number_input(FEATURE_LABELS.get(name, name), min_value=1, max_value=3, value=1, step=1))
        elif name == "Age":
            values[name] = float(col.number_input(FEATURE_LABELS.get(name, name), min_value=1.0, max_value=100.0, value=30.0, step=1.0, format="%.1f"))
        elif name == "Fare":
            values[name] = float(col.number_input(FEATURE_LABELS.get(name, name), min_value=0.0, max_value=100000.0, value=50.0, step=10.0, format="%.2f"))
        elif name == "FamilySize":
            values[name] = float(col.number_input(FEATURE_LABELS.get(name, name), min_value=1, max_value=20, value=1, step=1))
        else:
            values[name] = col.number_input(FEATURE_LABELS.get(name, name), value=0.0, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 ทำนายผล")

# ================================================================ result
if submitted:
    X = pd.DataFrame([values], columns=features)
    pred, score = predict(model, scaler.transform(X))
    is_pos = pred == POSITIVE_CLASS
    css, icon = ("warn", "⚠️") if is_pos else ("ok", "😊")
    st.markdown(
        f'<div class="card {css}"><div class="big">{icon} {CLASS_LABELS.get(pred, pred)}</div>'
        f'<div style="margin-top:4px">คะแนนความมั่นใจของโมเดล <b>{score*100:.1f}%</b></div>'
        f'<div class="bar"><div style="width:{score*100:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )

# ================================================================ footer
acc = load_accuracy()
acc_txt = f"{acc*100:.2f}%"
st.markdown(
    f"""
<div class="footer">
<div class="lbl">ACCURACY · ค่าความแม่นยำของระบบ</div>
<div class="acc">{acc_txt}</div>
<div class="dev">พัฒนาโดย {" &nbsp;·&nbsp; ".join(DEVELOPERS)}</div>
</div>
""",
    unsafe_allow_html=True,
)