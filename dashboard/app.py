# -*- coding: utf-8 -*-
import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def detect_anomaly(y_true, y_pred, threshold=0.5):
    error = np.abs(y_true - y_pred)
    if error > threshold:
        return True, error
    else:
        return False, error


from src.forecast import (
    load_available_models,
    load_model,
    build_features_from_history,
    recursive_forecast,
    add_acceleration_column,
    FEATURES,
)

st.set_page_config(page_title="Thruster Forecast Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 Thruster Forecast Dashboard — پیش‌بینی چند-گام جلوتر رانش/شتاب")

st.markdown("""
این داشبورد به شما اجازه می‌دهد **مدل XGBoost** پروژه را انتخاب کنید، چند ثانیه‌ی آینده را **پیش‌بینی** کنید
و در صورت وارد کردن **جرم فضاپیما (kg)**، شتاب تخمینی \\(a = F / m\\) را نیز ببینید.

**روش کار** (خلاصه):
1) یک **مدل** انتخاب کنید (سراسری یا مدل اختصاصی SN).
2) تاریخچه‌ی کوتاهی از داده‌ها را بارگذاری کنید (CSV) — حداقل ستون‌های `thrust`, `ton`, `on_duration`.
3) افق پیش‌بینی (تعداد گام) و اندازه‌ی گام زمانی (ثانیه) را تنظیم کنید.
4) (اختیاری) جرم فضاپیما را وارد کنید تا شتاب هم محاسبه شود.
""")

models_dir = st.sidebar.text_input("مسیر پوشه مدل‌ها", value="models")
available = load_available_models(models_dir)

if not available:
    st.error("هیچ مدلی در مسیر مشخص‌شده پیدا نشد. پوشه `models/` شامل `xgb_final_model.joblib` یا زیرپوشه `models/individual/` باید وجود داشته باشد.")
    st.stop()

model_label = st.sidebar.selectbox("انتخاب مدل", options=list(available.keys()))
model_path = available[model_label]

# Load model (cached)
@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return load_model(path)

model = _load_model_cached(model_path)

st.sidebar.markdown("---")
steps = st.sidebar.number_input("تعداد گام پیش‌بینی (seconds ahead)", min_value=1, max_value=120, value=10, step=1)
dt = st.sidebar.number_input("اندازه گام زمانی (ثانیه)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
future_on = st.sidebar.number_input("on_duration آینده (اگر خالی بگذارید، به صورت افزایشی در نظر گرفته می‌شود)", min_value=0.0, value=0.0, step=0.1)
use_future_on = st.sidebar.checkbox("استفاده از مقدار ثابت برای on_duration آینده؟", value=False)

mass_kg = st.sidebar.number_input("جرم فضاپیما (kg) — اختیاری", min_value=0.0, value=0.0, step=1.0)
mass_value = mass_kg if mass_kg > 0 else None

st.markdown("### 1) بارگذاری تاریخچه‌ی کوتاه (CSV)")
uploaded = st.file_uploader("فایلی با ستون‌های حداقل: thrust, ton, on_duration (ستون timestamp اختیاری)", type=["csv"])

example = st.toggle("نمونه‌ی مصنوعی بساز (اگر فایل نداری)")
if example and uploaded is None:
    # Make a tiny synthetic example history
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="S"),
        "thrust": np.linspace(10, 20, 10) + np.random.normal(0, 0.5, 10),
        "ton": np.arange(10).astype(float),
        "on_duration": np.ones(10).astype(float)
    }
    hist_df = pd.DataFrame(data)
else:
    if uploaded is None:
        st.info("یک CSV آپلود کنید یا گزینه‌ی ساخت نمونه‌ی مصنوعی را فعال کنید.")
        st.stop()
    hist_df = pd.read_csv(uploaded)

st.write("**بخشی از تاریخچه ورودی شما:**")
st.dataframe(hist_df.head(10))

st.markdown("### 2) محاسبه ویژگی‌ها و اعتبارسنجی ورودی")
try:
    with st.spinner("در حال محاسبه ویژگی‌ها..."):
        feat_df = build_features_from_history(hist_df, window=5)
        valid_df = feat_df.dropna(subset=['lag_thrust_1']).reset_index(drop=True)
        if valid_df.empty:
            st.error("برای ایجاد ویژگی lag به حداقل **دو ردیف** تاریخچه نیاز است.")
            st.stop()
        st.success("ویژگی‌ها ساخته شد ✅")
        st.dataframe(valid_df.tail(5))
except Exception as e:
    st.exception(e)
    st.stop()

st.markdown("### 3) پیش‌بینی چند-گام جلوتر")
apply_future_on = future_on if use_future_on else None
with st.spinner("در حال پیش‌بینی..."):
    fc = recursive_forecast(
        model=model,
        history_df=hist_df,
        steps=int(steps),
        step_seconds=float(dt),
        future_on_duration=apply_future_on,
        window=5
    )
    fc = add_acceleration_column(fc, mass_value)

left, right = st.columns([2,1])
with left:
    st.line_chart(fc.set_index("t_step")[["pred_thrust"]])
    if mass_value is not None and "pred_accel_mps2" in fc.columns:
        st.line_chart(fc.set_index("t_step")[["pred_accel_mps2"]])

with right:
    st.markdown("**خلاصه**")
    st.write({
        "مدل": model_label,
        "گام‌ها": int(steps),
        "Δt": float(dt),
        "mass_kg": mass_value
    })
    st.download_button(
        label="دانلود خروجی پیش‌بینی (CSV)",
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name="thruster_forecast.csv",
        mime="text/csv"
    )

st.header("🔍 Anomaly Detection - تشخیص رفتار غیرعادی موتور")

y_true = st.number_input("مقدار واقعی شتاب (از سنسور یا دیتاست):", value=0.0)
y_pred = st.number_input("مقدار پیش‌بینی‌شده توسط مدل:", value=0.0)
threshold = st.slider("آستانه خطا (Threshold):", 0.1, 5.0, 1.0)

anomaly, error = detect_anomaly(y_true, y_pred, threshold)

if anomaly:
    st.error(f"⚠️ هشدار: رفتار غیرعادی! اختلاف = {error:.2f}")
else:
    st.success(f"✅ وضعیت عادی است. اختلاف = {error:.2f}")

st.markdown("### 4) نکات مهم")
st.markdown("""
- ستون‌های حداقل لازم برای تاریخچه: `thrust`, `ton`, `on_duration` (اختیاری: `timestamp`).  
- این داشبورد بدون **Deep Learning** و با مدل‌های **XGBoost** از قبل آموزش‌دیده‌ی شما کار می‌کند.  
- اگر می‌خواهید خروجی شتاب را ببینید، **جرم فضاپیما** را در نوار کنار وارد کنید.
- برای سناریوهای آینده، می‌توانید `on_duration` را ثابت کنید یا بگذارید به صورت افزایشی جلو برود.
""")
