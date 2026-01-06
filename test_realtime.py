# =========================================================
# REAL-TIME INFERENCE PIPELINE (STUDENT MODEL)
# FINAL OUTPUT CSV COLUMNS:
# revision, timestamp, ghi, actual, prediction, error
# =========================================================

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load

from pvlib import solarposition
from pvlib.location import Location

from models.power_net import PowerNet
from data.data_utils import add_features
from config.config_mosdac import *

# =========================================================
# CONFIG
# =========================================================
REVISION_ROOT = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_020126")
SITE_CSV      = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_sites.csv")
POWER_DIR     = Path("/mnt/DATA1/pankhi/forecast/DATA/all_site_with_data")

MODEL_DIR = Path(SAVE_DIR)
OUT_DIR   = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/results_realtime")

TZ = "Asia/Kolkata"
DEVICE = DEVICE

# =========================================================
# UTILS
# =========================================================
def ensure_tz(series):
    s = pd.to_datetime(series)
    return s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)

def find_revision_csv(root: Path, site_name: str):
    key = site_name.lower().replace(" ", "_")
    for p in root.glob("*.csv"):
        if key in p.name.lower():
            return p
    return None

# =========================================================
# LOAD SITE METADATA
# =========================================================
sites = pd.read_csv(SITE_CSV)
sites.columns = [c.strip().lower() for c in sites.columns]
site = sites[sites["pos_name"] == SITE_NAME].iloc[0]

LAT, LON = site["latitude"], site["longitude"]

site_loc = Location(LAT, LON, tz=TZ)

# =========================================================
# LOAD POWER DATA (FOR ACTUAL + CAPACITY)
# =========================================================
power = pd.read_csv(POWER_DIR / f"{SITE_NAME}.csv")
power.columns = [c.strip().lower() for c in power.columns]

power["timestamp"] = ensure_tz(power["timestamp"]).dt.tz_localize(None)

cap_col = next(c for c in power.columns if "available" in c or "avc" in c)
power = power.rename(columns={cap_col: "available_capacity"})

LAST_CAPACITY = power["available_capacity"].max()

# =========================================================
# LOAD STUDENT MODEL + SCALER
# =========================================================
device = torch.device(DEVICE)

model = PowerNet(
    input_dim=len(STUDENT_FEATURES),
    hidden=HYPERPARAMS["hidden"],
    drop=HYPERPARAMS["dropout"]
).to(device)

model.load_state_dict(
    torch.load(MODEL_DIR / "best_student_powernet.pt", map_location=device)
)
model.eval()

scaler_x = load(MODEL_DIR / "scaler_x_student.pkl")

print("✅ Student model and scaler loaded")

# =========================================================
# MAIN REAL-TIME INFERENCE
# =========================================================
def run_realtime_inference():

    revision_csv = find_revision_csv(REVISION_ROOT, SITE_NAME)
    if revision_csv is None:
        raise FileNotFoundError("❌ Revision CSV not found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{revision_csv.stem}.csv"

    rev = pd.read_csv(revision_csv)
    rev.columns = [c.strip().lower() for c in rev.columns]

    # Authoritative timestamps from GHI CSV (IST)
    rev["timestamp"] = ensure_tz(rev["valid_time_ist"]).dt.tz_localize(None)
    rev = rev.sort_values(["revision", "timestamp"]).reset_index(drop=True)

    outputs = []

    for rev_id in rev["revision"].unique():
        batch = rev[rev["revision"] == rev_id].copy()
        print(f"▶ Processing {rev_id} | {len(batch)} rows")

        # ---------------- INPUTS ----------------
        batch["ghi"] = batch.get("ghi_pred_wm2", 0.0)

        batch["available_capacity"] = [
            power.loc[power["timestamp"] <= ts, "available_capacity"].max()
            if not power.loc[power["timestamp"] <= ts].empty
            else LAST_CAPACITY
            for ts in batch["timestamp"]
        ]

        # ---------------- SOLAR GEOMETRY ----------------
        solpos = solarposition.get_solarposition(
            ensure_tz(batch["timestamp"]), LAT, LON
        )
        batch["zenith"] = solpos["zenith"].values
        batch["cos_zenith"] = np.cos(np.radians(batch["zenith"]))

        # ---------------- CLEAR-SKY GHI (CRITICAL FIX) ----------------
        # pvlib REQUIRES tz-aware DatetimeIndex (NOT Series)
        ts_index = pd.DatetimeIndex(
            ensure_tz(batch["timestamp"]).values
        )

        clearsky = site_loc.get_clearsky(
            ts_index,
            model="ineichen"
        )

        batch["clear_sky_ghi"] = np.clip(
            clearsky["ghi"].values,
            1.0,
            None
        )

        # ---------------- FEATURE ENGINEERING ----------------
        batch["Timestamp"] = batch["timestamp"]
        batch = add_features(batch, mode="infer")
        batch.drop(columns=["Timestamp"], inplace=True, errors="ignore")

        # ---------------- SCALE ----------------
        X = batch[STUDENT_FEATURES].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill().fillna(0.0)

        x = scaler_x.transform(X)

        # ---------------- MODEL ----------------
        with torch.no_grad():
            cf_pred = model(
                torch.tensor(x, dtype=torch.float32).to(device)
            ).cpu().numpy().ravel()

        cf_pred = np.clip(cf_pred, 0.0, 1.2)

        batch["prediction"] = np.clip(
            cf_pred * batch["available_capacity"].values,
            0.0,
            batch["available_capacity"].values
        )

        # ---------------- ACTUAL (IF AVAILABLE) ----------------
        actual_vals = []
        for ts in batch["timestamp"]:
            row = power.loc[power["timestamp"] == ts]
            actual_vals.append(
                row["actual"].values[0] if not row.empty else np.nan
            )

        batch["actual"] = actual_vals

        # ---------------- ERROR ----------------
        batch["error"] = batch["prediction"] - batch["actual"]

        # ---------------- FINAL OUTPUT ----------------
        batch = batch[[
            "revision",
            "timestamp",
            "ghi",
            "actual",
            "prediction",
            "error"
        ]]

        outputs.append(batch)

    final_df = pd.concat(outputs, ignore_index=True)
    final_df.to_csv(out_csv, index=False)

    print(f"✅ Final CSV saved → {out_csv}")
    return final_df

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_realtime_inference()
