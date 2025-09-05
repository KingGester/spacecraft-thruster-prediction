# -*- coding: utf-8 -*-
"""
Utilities for loading models and making single-step and multi-step forecasts
for spacecraft thruster thrust, plus optional conversion to acceleration
given spacecraft mass.
"""
from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# Default features used during training (based on your training script)
FEATURES = ["ton", "on_duration", "lag_thrust_1", "rolling_avg_thrust", "cumulative_on_time"]

def load_available_models(models_dir: str) -> Dict[str, str]:
    """
    Scan a models directory and return a mapping {model_label: path}.
    - Global model is named 'xgb_final_model.joblib'
    - Per-SN models are in 'models/individual/xgb_model_{sn}.joblib'
    """
    models = {}
    global_model = os.path.join(models_dir, "xgb_final_model.joblib")
    if os.path.exists(global_model):
        models["Global XGB"] = global_model

    indiv_dir = os.path.join(models_dir, "individual")
    if os.path.isdir(indiv_dir):
        for fn in sorted(os.listdir(indiv_dir)):
            if fn.startswith("xgb_model_") and fn.endswith(".joblib"):
                path = os.path.join(indiv_dir, fn)
                # Extract SN from filename
                try:
                    sn = fn.replace("xgb_model_", "").replace(".joblib", "")
                    label = f"Per-SN XGB | SN {sn}"
                except Exception:
                    label = f"Per-SN XGB | {fn}"
                models[label] = path
    return models


def load_model(model_path: str):
    """Load a joblib-saved estimator (e.g., XGBRegressor)."""
    return joblib.load(model_path)


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Your file must include these columns: {required}")


def build_features_from_history(history: pd.DataFrame,
                                window: int = 5) -> pd.DataFrame:
    """
    Given a time-sorted dataframe with columns at least ['thrust','ton','on_duration'],
    compute derived feature columns used by the model:
    - lag_thrust_1: previous step thrust
    - rolling_avg_thrust: rolling mean of thrust with 'window' length
    - cumulative_on_time: cumulative sum of on_duration
    Returns a copy of the df with feature columns added.
    """
    df = history.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    ensure_required_columns(df, ["thrust", "ton", "on_duration"])

    df["lag_thrust_1"] = df["thrust"].shift(1)
    df["rolling_avg_thrust"] = (
        df["thrust"].rolling(window=window, min_periods=1).mean()
    )
    # If 'on_duration' is per-step duration, cumulative is cumsum
    df["cumulative_on_time"] = df["on_duration"].cumsum()
    return df


def single_step_predict(model, last_row: pd.Series) -> float:
    """
    Predict next-step thrust using the last available feature row.
    """
    X = last_row[FEATURES].values.reshape(1, -1)
    y_hat = float(model.predict(X)[0])
    return y_hat


def recursive_forecast(model,
                       history_df: pd.DataFrame,
                       steps: int = 5,
                       step_seconds: float = 1.0,
                       future_on_duration: Optional[float] = None,
                       window: int = 5) -> pd.DataFrame:
    """
    Make a multi-step forecast by iteratively predicting the next thrust and
    feeding it back as the new 'thrust'.

    Parameters
    ----------
    model : fitted estimator
    history_df : pd.DataFrame
        Must include columns ['thrust','ton','on_duration'] at minimum.
        Additional columns are ignored.
    steps : int
        Number of steps to forecast.
    step_seconds : float
        Delta time per step. 'ton' and 'on_duration' are incremented using this.
    future_on_duration : Optional[float]
        If provided, use this constant value for 'on_duration' at each future step.
        Otherwise we increment 'on_duration' by +step_seconds each step.
    window : int
        Window for rolling average.

    Returns
    -------
    forecast_df : pd.DataFrame
        Contains 't_step', 'pred_thrust' and cumulative time columns.
    """
    df = build_features_from_history(history_df, window=window).copy()
    # Ensure there is at least one valid row with no NaNs in FEATURES
    df = df.dropna(subset=["lag_thrust_1"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Not enough history to compute lag features. "
                         "Provide at least 2 rows of history.")

    # Start from the last observed row
    last = df.iloc[-1].copy()

    # Prepare forecast result collector
    out_rows = []
    base_ton = float(last["ton"])
    base_cum_on = float(last["cumulative_on_time"])

    for i in range(1, steps + 1):
        # Predict next thrust
        y_hat = single_step_predict(model, last)

        # Advance 'ton' and 'on_duration'
        ton_next = base_ton + i * step_seconds
        if future_on_duration is None:
            on_duration_next = last["on_duration"] + step_seconds
        else:
            on_duration_next = float(future_on_duration)

        # Update cumulative on-time
        cumulative_on_time_next = base_cum_on + i * step_seconds \
            if future_on_duration is None else base_cum_on + i * on_duration_next

        # Update rolling average: naive update using last 'window-1' values + new y_hat
        # Here we just use the previous rolling avg and incorporate the new prediction.
        # For simplicity, we recompute from a small buffer when possible.
        # Maintain a small deque-like list of recent thrusts:
        # (In practice, for performance you'd keep a deque; here it's okay.)
        if "recent_thrusts" not in locals():
            recent_thrusts = list(df["thrust"].tail(window).values.astype(float))
        recent_thrusts.append(y_hat)
        if len(recent_thrusts) > window:
            recent_thrusts = recent_thrusts[-window:]
        rolling_avg_thrust_next = float(np.mean(recent_thrusts))

        # Build next feature row
        next_row = pd.Series({
            "ton": ton_next,
            "on_duration": on_duration_next,
            "lag_thrust_1": y_hat,  # previous thrust is now y_hat
            "rolling_avg_thrust": rolling_avg_thrust_next,
            "cumulative_on_time": cumulative_on_time_next,
        })

        out_rows.append({
            "t_step": i,
            "ton": ton_next,
            "on_duration": on_duration_next,
            "cumulative_on_time": cumulative_on_time_next,
            "pred_thrust": y_hat,
        })

        # The 'last' feature row for the next iteration
        last = next_row.copy()

    forecast_df = pd.DataFrame(out_rows)
    return forecast_df


def thrust_to_acceleration(thrust_newtons: float, mass_kg: float) -> Optional[float]:
    """
    Convert thrust (N) to acceleration (m/s^2) using a = F / m.
    Returns None if mass is invalid.
    """
    if mass_kg is None or mass_kg <= 0:
        return None
    return float(thrust_newtons / mass_kg)


def add_acceleration_column(df: pd.DataFrame, mass_kg: Optional[float]) -> pd.DataFrame:
    """
    If mass_kg is provided (>0), add 'pred_accel_mps2' = pred_thrust / mass_kg.
    """
    out = df.copy()
    if mass_kg is not None and mass_kg > 0 and "pred_thrust" in out.columns:
        out["pred_accel_mps2"] = out["pred_thrust"].astype(float) / float(mass_kg)
    return out
