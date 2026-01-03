# =========================================================
# DATA UTILS: Load, Clean, Feature Engineering & Scaling
# =========================================================
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
from joblib import dump
import os

# ---------- 1️⃣ Load CSV ----------
def load_power(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    time_col = next((c for c in df.columns if "timestamp" in c or "time" in c), df.columns[0])
    ghi_col = next((c for c in df.columns if "ghi" in c.lower()), None)
    power_col = next((c for c in df.columns if "actual" in c.lower() or "power" in c.lower()), None)
    zenith_col = next((c for c in df.columns if c.lower() == "zenith"), None)
    cos_col = next((c for c in df.columns if c.lower() == "cos_zenith"), None)

    # EXACT temperature column t2m
    temp_col = "t2m" if "t2m" in df.columns else None

    if not all([time_col, ghi_col, power_col, zenith_col, cos_col]):
        raise ValueError(f"❌ Missing key columns in {csv_path}")

    rename_dict = {
        time_col: "Timestamp",
        ghi_col: "ghi",
        power_col: "Actual",
        zenith_col: "zenith",
        cos_col: "cos_zenith"
    }

    df.rename(columns=rename_dict, inplace=True)
    return df




# ---------- Feature Engineering ----------
def add_features(df, mode="train", rng=None):
    """
    mode: "train" or "infer"
    rng : np.random.Generator (optional, for reproducibility)
    """

    df = df.copy()

    if rng is None:
        rng = np.random.default_rng()

    # =====================================================
    # GHI AUGMENTATION (TRAINING ONLY)
    # =====================================================
    if mode == "train":
        # --- noise level mimics different revisions ---
        sigma = rng.choice([0.05, 0.10, 0.15])  # late → early revisions

        noise = rng.normal(0.0, sigma, size=len(df))

        # --- multiplicative bias ---
        bias = rng.uniform(0.9, 1.1)

        ghi_aug = df["ghi"] * (1.0 + noise) * bias

        # physical constraints
        ghi_aug = ghi_aug.clip(lower=0.0)

        df["ghi_aug"] = ghi_aug

    else:
        # inference: NO augmentation
        df["ghi_aug"] = df["ghi"]

    # =====================================================
    # GHI DERIVED FEATURES (FROM ghi_aug)
    # =====================================================
    df["log_GHI"]   = np.log1p(df["ghi_aug"])
    df["GHI_sq"]    = df["ghi_aug"] ** 2
    df["GHI_cubed"] = df["ghi_aug"] ** 3
    df["GHI_cosZ"]  = df["ghi_aug"] * df["cos_zenith"]

    # =====================================================
    # TIME FEATURES
    # =====================================================
    ts = pd.to_datetime(df["Timestamp"])
    df["hour"] = ts.dt.hour + ts.dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # =====================================================
    # TEMPERATURE (RAW)
    # =====================================================
    if "t2m" in df.columns:
        df["t2m"] = df["t2m"].astype(float)

    return df


# ---------- 4️⃣ Feature Scaling ----------
def scale_features(df_train, df_val, df_test, feature_cols, save_dir):
    scaler = RobustScaler()
    x_train = scaler.fit_transform(df_train[feature_cols])
    x_val   = scaler.transform(df_val[feature_cols])
    x_test  = scaler.transform(df_test[feature_cols])

    dump({"scaler_x": scaler}, os.path.join(save_dir, "scaler_x.pkl"))
    return x_train, x_val, x_test


# ---------- 5️⃣ Preprocess Data: Bias Normalization + Target ----------
def preprocess_data(df_train, df_val, df_test, save_dir):

    # Power + GHI normalization constants
    P_mean, P_std = df_train["Actual"].mean(), df_train["Actual"].std()
    GHI_mean, GHI_std = df_train["ghi"].mean(), df_train["ghi"].std()

    for df in [df_train, df_val, df_test]:
        df["ghi_norm"] = (df["ghi"] - GHI_mean) / (GHI_std + 1e-6)
        df["power_norm"] = (df["Actual"] - P_mean) / (P_std + 1e-6)

        df["y_rel"] = (df["power_norm"] / (df["ghi_norm"] + 1e-6)) - 1
        df["y_rel"] = df["y_rel"].clip(-0.5, 1.0)

    # Save constants
    np.save(
        os.path.join(save_dir, "power_ghi_mean_std.npy"),
        np.array([P_mean, P_std, GHI_mean, GHI_std])
    )

    print("📦 Saved normalization constants.")

    return df_train, df_val, df_test
