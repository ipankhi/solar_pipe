# =========================================================
# DATA UTILS: Load, Feature Engineering & Scaling
# (Prediction setup: GHI → Power)
# =========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from joblib import dump
import os

# =========================================================
# 1️⃣ Load CSV
# =========================================================
def load_power(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    time_col = next((c for c in df.columns if "timestamp" in c or c == "time"), None)
    if time_col is None:
        raise ValueError(f"❌ Timestamp column missing in {csv_path}")

    df = df.rename(columns={time_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df


# =========================================================
# 2️⃣ Feature Engineering
# =========================================================
def add_features(df, mode="train"):
    """
    mode:
      - 'train' / 'eval' → compute cf (needs actual)
      - 'infer'          → NO cf, NO actual dependency
    """

    assert mode in {"train", "eval", "infer"}

    df = df.copy()

    # ---------- Required physics ----------
    if "clear_sky_ghi" not in df.columns:
        raise ValueError("❌ clear_sky_ghi missing (must be precomputed)")

    if "ghi" not in df.columns:
        raise ValueError("❌ ghi missing")

    if "available_capacity" not in df.columns:
        raise ValueError("❌ available_capacity missing")

    # ---------- Physics-derived ----------
    df["csi"] = df["ghi"] / (df["clear_sky_ghi"] + 1e-6)
    df["csi"] = df["csi"].clip(0.0, 1.5)

    df["ghi_sq"] = df["ghi"] ** 2
    df["ghi_cu"] = df["ghi"] ** 3

    # ---------- Temperature (teacher only) ----------
    if "t2m" in df.columns:
        df["t2m"] = df["t2m"].astype(float)

    # ---------- Target (ONLY for train / eval) ----------
    if mode in {"train", "eval"}:
        if "actual" not in df.columns:
            raise ValueError("❌ actual missing for training/eval")

        df["cf"] = df["actual"] / (df["available_capacity"] + 1e-6)
        df["cf"] = df["cf"].clip(0.0, 1.2)

    return df


# =========================================================
# 3️⃣ Feature Scaling
# =========================================================
def scale_features(mode, df_train, df_val, df_test, feature_cols, save_dir):
    """
    mode: 'teacher' or 'student'
    """

    scaler = RobustScaler()

    x_train = scaler.fit_transform(df_train[feature_cols])
    x_val   = scaler.transform(df_val[feature_cols])
    x_test  = scaler.transform(df_test[feature_cols])

    os.makedirs(save_dir, exist_ok=True)
    dump(
        scaler,
        os.path.join(save_dir, f"scaler_x_{mode}.pkl")
    )

    return x_train, x_val, x_test
