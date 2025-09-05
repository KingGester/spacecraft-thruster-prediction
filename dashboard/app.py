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

from src.forecast import (
    load_available_models,
    load_model,
    build_features_from_history,
    recursive_forecast,
    add_acceleration_column,
    FEATURES,
)

# ---------------- Language dictionary ----------------
translations = {
    "fa": {
        "title": "🚀 داشبورد پیش‌بینی رانش/شتاب چند-گام جلوتر",
        "intro": """
این داشبورد به شما اجازه می‌دهد **مدل XGBoost** پروژه را انتخاب کنید، چند ثانیه‌ی آینده را **پیش‌بینی** کنید  
و در صورت وارد کردن **جرم فضاپیما (kg)**، شتاب تخمینی \\(a = F / m\\) را نیز ببینید.

**روش کار (خلاصه):**
1) یک **مدل** انتخاب کنید (سراسری یا مدل اختصاصی SN).  
2) تاریخچه‌ی کوتاهی از داده‌ها را بارگذاری کنید (CSV) — حداقل ستون‌های `thrust`, `ton`, `on_duration`.  
3) افق پیش‌بینی (تعداد گام) و اندازه‌ی گام زمانی (ثانیه) را تنظیم کنید.  
4) (اختیاری) جرم فضاپیما را وارد کنید تا شتاب هم محاسبه شود.  
        """,
        "models_dir": "مسیر پوشه مدل‌ها",
        "model_select": "انتخاب مدل",
        "steps": "تعداد گام پیش‌بینی (seconds ahead)",
        "dt": "اندازه گام زمانی (ثانیه)",
        "future_on": "on_duration آینده (اگر خالی بگذارید، به صورت افزایشی در نظر گرفته می‌شود)",
        "use_future_on": "استفاده از مقدار ثابت برای on_duration آینده؟",
        "mass": "جرم فضاپیما (kg) — اختیاری",
        "step1": "### 1) بارگذاری تاریخچه‌ی کوتاه (CSV)",
        "generate_example": "نمونه‌ی مصنوعی بساز (اگر فایل نداری)",
        "sample_history": "**بخشی از تاریخچه ورودی شما:**",
        "step2": "### 2) محاسبه ویژگی‌ها و اعتبارسنجی ورودی",
        "step3": "### 3) پیش‌بینی چند-گام جلوتر",
        "summary": "**خلاصه**",
        "download": "دانلود خروجی پیش‌بینی (CSV)",
        "anomaly_title": "🔍 تشخیص رفتار غیرعادی موتور",
        "y_true": "مقدار واقعی شتاب (از سنسور یا دیتاست):",
        "y_pred": "مقدار پیش‌بینی‌شده توسط مدل:",
        "threshold": "آستانه خطا:",
        "anomaly_warn": "⚠️ هشدار: رفتار غیرعادی! اختلاف = {error:.2f}",
        "anomaly_ok": "✅ وضعیت عادی است. اختلاف = {error:.2f}",
        "notes": "### 4) نکات مهم",
        "notes_text": """
- ستون‌های حداقل لازم برای تاریخچه: `thrust`, `ton`, `on_duration` (اختیاری: `timestamp`).  
- این داشبورد بدون **Deep Learning** و با مدل‌های **XGBoost** از قبل آموزش‌دیده‌ی شما کار می‌کند.  
- اگر می‌خواهید خروجی شتاب را ببینید، **جرم فضاپیما** را در نوار کنار وارد کنید.  
- برای سناریوهای آینده، می‌توانید `on_duration` را ثابت کنید یا بگذارید به صورت افزایشی جلو برود.  
        """,
    },
    "en": {
        "title": "🚀 Thruster Forecast Dashboard — Multi-step Thrust/Acceleration Prediction",
        "intro": """
This dashboard allows you to **select the XGBoost model**, forecast several seconds into the **future**,  
and if you provide the **spacecraft mass (kg)**, it will also show the estimated acceleration \\(a = F / m\\).

**Workflow (summary):**
1) Select a **model** (global or individual SN model).  
2) Upload a short history dataset (CSV) — must include at least `thrust`, `ton`, `on_duration`.  
3) Configure the forecast horizon (number of steps) and time step size (seconds).  
4) (Optional) Enter spacecraft mass to also compute acceleration.  
        """,
        "models_dir": "Models directory path",
        "model_select": "Select model",
        "steps": "Forecast steps (seconds ahead)",
        "dt": "Time step size (seconds)",
        "future_on": "Future on_duration (leave empty for incremental)",
        "use_future_on": "Use fixed value for future on_duration?",
        "mass": "Spacecraft mass (kg) — optional",
        "step1": "### 1) Upload short history (CSV)",
        "generate_example": "Generate synthetic example (if no file available)",
        "sample_history": "**Sample of your input history:**",
        "step2": "### 2) Feature calculation and input validation",
        "step3": "### 3) Multi-step forecast",
        "summary": "**Summary**",
        "download": "Download forecast output (CSV)",
        "anomaly_title": "🔍 Anomaly Detection — Thruster abnormal behavior",
        "y_true": "Actual acceleration value (from sensor or dataset):",
        "y_pred": "Predicted value by the model:",
        "threshold": "Error threshold:",
        "anomaly_warn": "⚠️ Warning: Anomaly detected! Difference = {error:.2f}",
        "anomaly_ok": "✅ Normal condition. Difference = {error:.2f}",
        "notes": "### 4) Key Notes",
        "notes_text": """
- Minimum required columns in the history: `thrust`, `ton`, `on_duration` (optional: `timestamp`).  
- This dashboard works with your pre-trained **XGBoost models** (no Deep Learning here).  
- To see acceleration output, enter the **spacecraft mass** in the sidebar.  
- For future scenarios, you can either keep `on_duration` fixed or let it increase step by step.  
        """,
    }
}

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="Thruster Forecast Dashboard", page_icon="🚀", layout="wide")

# ---------------- Sidebar: language selector ----------------
lang = st.sidebar.radio("Language / زبان", ["fa", "en"], index=0)
T = translations[lang]

# ---------------- UI ----------------
st.title(T["title"])
st.markdown(T["intro"])

models_dir = st.sidebar.text_input(T["models_dir"], value="models")
available = load_available_models(models_dir)

if not available:
    st.error("No models found in the specified path. Ensure `models/` contains `xgb_final_model.joblib` or `models/individual/`.")
    st.stop()

model_label = st.sidebar.selectbox(T["model_select"], options=list(available.keys()))
model_path = available[model_label]

@st.cache_resource(show_spinner=False)
def _load_model_cached(path: str):
    return load_model(path)

model = _load_model_cached(model_path)

st.sidebar.markdown("---")
steps = st.sidebar.number_input(T["steps"], min_value=1, max_value=120, value=10, step=1)
dt = st.sidebar.number_input(T["dt"], min_value=0.1, max_value=10.0, value=1.0, step=0.1)
future_on = st.sidebar.number_input(T["future_on"], min_value=0.0, value=0.0, step=0.1)
use_future_on = st.sidebar.checkbox(T["use_future_on"], value=False)

mass_kg = st.sidebar.number_input(T["mass"], min_value=0.0, value=0.0, step=1.0)
mass_value = mass_kg if mass_kg > 0 else None

st.markdown(T["step1"])
uploaded = st.file_uploader("CSV file", type=["csv"])

example = st.toggle(T["generate_example"])
if example and uploaded is None:
    data = {
        "timestamp": pd.date_range("2025-01-01", periods=10, freq="S"),
        "thrust": np.linspace(10, 20, 10) + np.random.normal(0, 0.5, 10),
        "ton": np.arange(10).astype(float),
        "on_duration": np.ones(10).astype(float)
    }
    hist_df = pd.DataFrame(data)
else:
    if uploaded is None:
        st.info("Upload a CSV or enable synthetic example.")
        st.stop()
    hist_df = pd.read_csv(uploaded)

st.write(T["sample_history"])
st.dataframe(hist_df.head(10))

st.markdown(T["step2"])
try:
    with st.spinner("Building features..."):
        feat_df = build_features_from_history(hist_df, window=5)
        valid_df = feat_df.dropna(subset=['lag_thrust_1']).reset_index(drop=True)
        if valid_df.empty:
            st.error("At least two rows of history are required for lag features.")
            st.stop()
        st.success("Features built ✅")
        st.dataframe(valid_df.tail(5))
except Exception as e:
    st.exception(e)
    st.stop()

st.markdown(T["step3"])
apply_future_on = future_on if use_future_on else None
with st.spinner("Forecasting..."):
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
    st.line_chart(fc.set_index("t_step")["pred_thrust"])
    if mass_value is not None and "pred_accel_mps2" in fc.columns:
        st.line_chart(fc.set_index("t_step")["pred_accel_mps2"])

with right:
    st.markdown(T["summary"])
    st.write({
        "model": model_label,
        "steps": int(steps),
        "Δt": float(dt),
        "mass_kg": mass_value
    })
    st.download_button(
        label=T["download"],
        data=fc.to_csv(index=False).encode("utf-8"),
        file_name="thruster_forecast.csv",
        mime="text/csv"
    )

st.header(T["anomaly_title"])
y_true = st.number_input(T["y_true"], value=0.0)
y_pred = st.number_input(T["y_pred"], value=0.0)
threshold = st.slider(T["threshold"], 0.1, 5.0, 1.0)

error = abs(y_true - y_pred)
if error > threshold:
    st.error(T["anomaly_warn"].format(error=error))
else:
    st.success(T["anomaly_ok"].format(error=error))

st.markdown(T["notes"])
st.markdown(T["notes_text"])
