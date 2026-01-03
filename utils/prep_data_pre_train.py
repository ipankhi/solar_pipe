import pandas as pd
import numpy as np
from pathlib import Path
from pvlib import solarposition

# =========================================================
# PATHS
# =========================================================
SITE_CSV = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_sites.csv")

GHI_ROOT  = Path("/mnt/DATA1/pankhi/forecast/DATA/GHI_allsite_csvs")
ERA5_ROOT = Path("/mnt/DATA1/pankhi/forecast/ERA5/interpolated")
POWER_DIR = Path("/mnt/DATA1/pankhi/forecast/DATA/all_site_with_data")

OUT_DIR = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TZ = "Asia/Kolkata"

# =========================================================
# READ SITE METADATA
# =========================================================
sites_df = pd.read_csv(SITE_CSV)
sites_df.columns = [c.strip().lower() for c in sites_df.columns]

assert {
    "pos_name", "state", "latitude", "longitude"
}.issubset(sites_df.columns)

# =========================================================
# HELPERS
# =========================================================
def normalize_underscore(name: str) -> str:
    return "_".join(str(name).strip().split())

def normalize_space(name: str) -> str:
    return " ".join(str(name).strip().split())

def ghi_filename(pos):
    return f"GHI_{normalize_underscore(pos)}.csv"

def era5_filename(pos):
    return f"{normalize_space(pos)}.csv"

def power_filename(pos):
    return f"{normalize_underscore(pos)}.csv"

def read_csv_safe(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def find_ghi_file(root: Path, pos: str):
    matches = list(root.rglob(ghi_filename(pos)))
    return matches[0] if matches else None

# =========================================================
# MAIN LOOP
# =========================================================
for _, row in sites_df.iterrows():

    pos   = row["pos_name"]
    state = row["state"]
    lat   = row["latitude"]
    lon   = row["longitude"]

    print(f"\n🔹 Processing {pos} ({state})")

    ghi_file   = find_ghi_file(GHI_ROOT, pos)
    era5_file  = ERA5_ROOT / state / era5_filename(pos)
    power_file = POWER_DIR / power_filename(pos)

    if ghi_file is None or not era5_file.exists() or not power_file.exists():
        print("  ❌ Missing input file(s), skipping")
        continue

    # ---------------- READ ----------------
    ghi   = read_csv_safe(ghi_file)
    era5  = read_csv_safe(era5_file)
    power = read_csv_safe(power_file)

    # ---------------- GHI: UTC → IST ----------------
    ghi_ts_col = "time" if "time" in ghi.columns else "timestamp"

    ghi["timestamp"] = pd.to_datetime(
        ghi[ghi_ts_col],
        utc=True,
        errors="coerce"
    )

    ghi["timestamp"] = (
        ghi["timestamp"]
        .dt.tz_convert(TZ)
        .dt.tz_localize(None)
    )

    # ---------------- POWER & ERA5 (already IST) ----------------
    power["timestamp"] = pd.to_datetime(power["timestamp"])
    era5["time"]       = pd.to_datetime(era5["time"])

    # ---------------- AVAILABLE CAPACITY ----------------
    cap_col = next(
        (c for c in power.columns if ("available" in c and "cap" in c) or ("avc" in c)),
        None
    )
    if cap_col is None:
        print("  ⚠️ Available capacity missing, skipping")
        continue

    power = power.rename(columns={cap_col: "available_capacity"})

    # ---------------- COMMON PERIOD ----------------
    start = max(ghi["timestamp"].min(), power["timestamp"].min(), era5["time"].min())
    end   = min(ghi["timestamp"].max(), power["timestamp"].max(), era5["time"].max())

    if start >= end:
        print("  ❌ No overlapping time range")
        continue

    # ---------------- 15-MIN GRID ----------------
    full_time = pd.date_range(start, end, freq="15min")

    ghi   = ghi.set_index("timestamp").reindex(full_time)
    power = power.set_index("timestamp").reindex(full_time)
    era5  = era5.set_index("time").reindex(full_time).interpolate("time")

    power["available_capacity"] = power["available_capacity"].ffill()

    # ---------------- BASE MERGE ----------------
    final = pd.DataFrame({
        "timestamp": full_time,
        "ghi": ghi["ghi"],
        "actual": power["actual"],
        "available_capacity": power["available_capacity"],
        "t2m": era5["t2m"]
    }).dropna(subset=["ghi", "actual", "t2m"])

    if final.empty:
        print("  ❌ Empty after merge")
        continue

    # =====================================================
    # TIME FEATURES
    # =====================================================
    final["hour"]  = final["timestamp"].dt.hour + final["timestamp"].dt.minute / 60
    final["month"] = final["timestamp"].dt.month

    # =====================================================
    # SOLAR GEOMETRY
    # =====================================================
    ts_local = final["timestamp"].dt.tz_localize(TZ, nonexistent="shift_forward")

    solpos = solarposition.get_solarposition(
        time=ts_local,
        latitude=lat,
        longitude=lon
    )

    final["zenith"] = solpos["zenith"].values
    final["cos_zenith"] = np.cos(np.radians(final["zenith"]))

    # =====================================================
    # CAPACITY FACTOR
    # =====================================================
    final["cf"] = final["actual"] / final["available_capacity"]
    final["cf"] = final["cf"].clip(0, 1.2)

    # =====================================================
    # CLIMATOLOGY (MONTH × HOUR)
    # =====================================================
    clim_src = final[
        (final["ghi"] > 20) &
        (final["cos_zenith"] > 0.05)
    ]

    clim = (
        clim_src
        .groupby(["month", "hour"])["cf"]
        .mean()
        .reset_index()
        .rename(columns={"cf": "climatology_cf"})
    )

    final = final.merge(clim, on=["month", "hour"], how="left")
    final["climatology_cf"] = final["climatology_cf"].fillna(0.0)
    final["climatology_power"] = (
        final["climatology_cf"] * final["available_capacity"]
    )

    # =====================================================
    # SAVE
    # =====================================================
    out_file = OUT_DIR / f"{normalize_underscore(pos)}.csv"
    final.to_csv(out_file, index=False)

    print(f"  ✅ Saved → {out_file}")
    print(f"     Rows: {len(final)} | "
          f"{final['timestamp'].iloc[0]} → {final['timestamp'].iloc[-1]}")

print("\n🎯 ALL SITES PROCESSED SUCCESSFULLY")
