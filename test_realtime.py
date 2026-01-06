# =========================================================
# REAL-TIME INFERENCE PIPELINE
# ALL REVISIONS → SINGLE CSV + SINGLE PLOT (DAILY)
# OUTPUT NAME = REVISION CSV NAME (e.g. Fermi_Solar_132kV_020126)
# =========================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from pvlib import solarposition

from models.power_net import PowerNet
from data.data_utils import add_features
from config.config_mosdac import *

# =========================================================
# CONFIG
# =========================================================
REVISION_ROOT = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_020126")
SITE_CSV      = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_sites.csv")
POWER_DIR     = Path("/mnt/DATA1/pankhi/forecast/DATA/all_site_with_data")

MODEL_DIR   = Path(SAVE_DIR)
RESULTS_ROOT = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/results_realtime")

TZ = "Asia/Kolkata"
DEVICE = DEVICE

# =========================================================
# UTILS
# =========================================================
def ensure_tz(series):
    s = pd.to_datetime(series)
    return s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)

def find_revision_csv(root: Path, site_name: str):
    site_name = site_name.lower()
    for p in root.glob("*.csv"):
        if site_name in p.name.lower():
            return p
    return None

# =========================================================
# LOAD SITE METADATA
# =========================================================
sites = pd.read_csv(SITE_CSV)
sites.columns = [c.strip().lower() for c in sites.columns]
site = sites[sites["pos_name"] == SITE_NAME].iloc[0]

LAT, LON = site["latitude"], site["longitude"]
STATE_NAME = site["state"].strip().replace(" ", "_").lower()
SITE_NAME_SAFE = SITE_NAME.replace(" ", "_")

SITE_OUT_DIR = RESULTS_ROOT / STATE_NAME / SITE_NAME_SAFE
PLOT_DIR = SITE_OUT_DIR / "plots"
SITE_OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD POWER DATA → CLIMATOLOGY (15-MIN BINS)
# =========================================================
power = pd.read_csv(POWER_DIR / f"{SITE_NAME}.csv")
power.columns = [c.strip().lower() for c in power.columns]
power["timestamp"] = ensure_tz(power["timestamp"]).dt.tz_localize(None)

cap_col = next(c for c in power.columns if "available" in c or "avc" in c)
power = power.rename(columns={cap_col: "available_capacity"})
LAST_CAPACITY = power["available_capacity"].max()

power["month"] = power["timestamp"].dt.month
power["hour"]  = power["timestamp"].dt.hour + (power["timestamp"].dt.minute // 15) * 0.25
power["cf"]    = power["actual"] / power["available_capacity"]

site_clim = (
    power.groupby(["month", "hour"])["cf"]
    .mean()
    .reset_index()
    .rename(columns={"cf": "climatology_cf"})
)

# =========================================================
# LOAD MODEL
# =========================================================
device = torch.device(DEVICE)

model = PowerNet(
    input_dim=len(FEATURE_COLS),
    hidden=HYPERPARAMS["hidden"],
    drop=HYPERPARAMS["dropout"],
)
model.load_state_dict(
    torch.load(MODEL_DIR / "best_powernet.pt", map_location=device)
)
model.to(device).eval()

scaler_x = load(MODEL_DIR / "scaler_x.pkl")["scaler_x"]

# =========================================================
# MAIN INFERENCE
# =========================================================
def run_inference_all():

    revision_csv = find_revision_csv(REVISION_ROOT, SITE_NAME)
    if revision_csv is None:
        raise FileNotFoundError("Revision CSV not found")

    # 🔑 output base name EXACTLY from revision csv
    out_base = revision_csv.stem
    OUT_CSV  = SITE_OUT_DIR / f"{out_base}.csv"
    OUT_PLOT = PLOT_DIR / f"{out_base}.png"

    rev = pd.read_csv(revision_csv)
    rev.columns = [c.strip().lower() for c in rev.columns]

    rev["timestamp"] = ensure_tz(rev["valid_time_ist"]).dt.tz_localize(None)
    rev = rev.sort_values(["revision", "timestamp"]).reset_index(drop=True)

    all_batches = []

    for rev_id in rev["revision"].unique():
        batch = rev[rev["revision"] == rev_id].copy()
        print(f"▶ Processing {rev_id} ({len(batch)} rows)")

        batch["run_t0_ist"] = batch["valid_time_ist"]
        batch["run_t0_utc"] = batch["valid_time_utc"]

        batch["ghi"] = batch.get("ghi_pred_wm2", 0.0)

        batch["Available_Capacity"] = [
            power.loc[power["timestamp"] <= ts, "available_capacity"].max()
            if not power.loc[power["timestamp"] <= ts].empty
            else LAST_CAPACITY
            for ts in batch["timestamp"]
        ]

        solpos = solarposition.get_solarposition(
            ensure_tz(batch["timestamp"]), LAT, LON
        )
        batch["zenith"] = solpos["zenith"].values
        batch["cos_zenith"] = np.cos(np.radians(batch["zenith"]))

        batch["month"] = batch["timestamp"].dt.month
        batch["hour"]  = batch["timestamp"].dt.hour + (batch["timestamp"].dt.minute // 15) * 0.25

        batch = batch.merge(site_clim, on=["month", "hour"], how="left")
        batch["climatology_cf"] = batch["climatology_cf"].fillna(0.0)
        batch["climatology_power"] = (
            batch["climatology_cf"] * batch["Available_Capacity"]
        )

        # ---------- FEATURES ----------
        batch["Timestamp"] = batch["timestamp"]
        ts_copy = batch["timestamp"].to_numpy().copy()
        batch = add_features(batch, mode="infer")
        batch["timestamp"] = ts_copy
        batch.drop(columns=["Timestamp"], inplace=True, errors="ignore")

        # ---------- SANITIZE ----------
        X = batch[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0.0)
        x = scaler_x.transform(X)

        with torch.no_grad():
            y_hat = model(
                torch.tensor(x, dtype=torch.float32).to(device)
            ).cpu().numpy().ravel()

        batch["Predicted_Power"] = (
            batch["climatology_power"] * (1.0 + y_hat)
        ).clip(lower=0.0)

        all_batches.append(batch)

    final_df = pd.concat(all_batches, ignore_index=True)

    final_df["revision_num"] = final_df["revision"].str.extract(r"(\d+)").astype(int)
    final_df = final_df.sort_values("revision_num").drop(columns="revision_num")

    final_df.to_csv(OUT_CSV, index=False)
    print(f"✅ CSV saved → {OUT_CSV}")

    return final_df, OUT_PLOT

# =========================================================
# PLOTTING
# =========================================================
def plot_daily(df, out_plot):

    plt.figure(figsize=(12, 5))
    for rev_id, g in df.groupby("revision"):
        plt.plot(g["timestamp"], g["Predicted_Power"], label=rev_id)

    plt.title(f"{SITE_NAME} | Daily Forecast (All Revisions)")
    plt.xlabel("Forecast Time")
    plt.ylabel("Predicted Power")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"📊 Plot saved → {out_plot}")

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    df, out_plot = run_inference_all()
    plot_daily(df, out_plot)
