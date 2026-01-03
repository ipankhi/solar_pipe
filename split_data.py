!pip install pvlib
import os
import pandas as pd
import numpy as np
from pvlib import solarposition

# ========= USER PATHS =========
x_dir = "/content/drive/MyDrive/phd_data/x_values_ist"          # full weather features (already IST)
ghi_dir = "/content/drive/MyDrive/phd_data/mosdac_ghi_site"     # only GHI (UTC, 15-min)
y_dir = "/content/drive/MyDrive/phd_data/site_wise_y_ist"       # power data (IST)
save_dir = "/content/drive/MyDrive/phd_data/splits_mosdac"
os.makedirs(save_dir, exist_ok=True)

# ========= SITE COORDINATES (lat, lon) =========
site_coords = {
    "Arinsun_RUMS": (24.5000005, 81.5800005),
    "ESPL_RSP": (24.53581944, 71.27088056),
    "GANDHAR_SOLAR": (21.82194444, 73.11472222),
    "GIPCL_RSP": (24.532, 71.253),
    "GSECL_RSP": (24.5165, 71.2525),
    "KAWAS_SOLAR": (21.17561, 72.68362),
    "Mahindra_RUMS": (24.477, 81.55329),
    "TPREL_RSP": (24.52, 71.3),
}

tz_ist = "Asia/Kolkata"

# ========= MAIN PROCESS FUNCTION =========
def process_site(site_name, lat, lon):
    print(f"\n🔹 Processing {site_name} ...")

    # -------- File paths --------
    x_path = os.path.join(x_dir, f"{site_name}_merged_ERA5_hourly.csv")
    ghi_path = os.path.join(ghi_dir, f"GHI_{site_name}_2022_2023.csv")
    y_path = os.path.join(y_dir, f"{site_name}_merged.csv")

    if not (os.path.exists(x_path) and os.path.exists(ghi_path) and os.path.exists(y_path)):
        print(f"⚠️ Missing data for {site_name}, skipping.")
        return

    # -------- Load CSVs --------
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
    df_ghi = pd.read_csv(ghi_path)

    # -------- Drop old GHI from X --------
    old_ghi_cols = [c for c in df_x.columns if "ghi" in c.lower()]
    if len(old_ghi_cols) > 0:
        print(f"🧹 Dropping old GHI columns from X: {old_ghi_cols}")
        df_x.drop(columns=old_ghi_cols, inplace=True)

    # -------- Parse timestamps --------
    # X values: assume already IST
    df_x["time"] = pd.to_datetime(df_x["time"], errors="coerce")

    # Y values: IST
    df_y["Timestamp"] = pd.to_datetime(df_y["Timestamp"], errors="coerce")

    # GHI values: UTC → IST
    df_ghi["Timestamp"] = pd.to_datetime(df_ghi["time"], errors="coerce", utc=True)
    df_ghi["Timestamp"] = df_ghi["Timestamp"].dt.tz_convert(tz_ist).dt.tz_localize(None)

    # keep only 2022–2023 period
    start_date, end_date = "2022-01-01", "2023-12-31 23:59:59"
    df_ghi = df_ghi[(df_ghi["Timestamp"] >= start_date) & (df_ghi["Timestamp"] <= end_date)]

    # try to locate ghi column
    ghi_col = [c for c in df_ghi.columns if "ghi" in c.lower()]
    if len(ghi_col) == 0:
        raise ValueError(f"No GHI column found in {ghi_path}")
    df_ghi = df_ghi[["Timestamp", ghi_col[0]]].rename(columns={ghi_col[0]: "GHI"})

    # -------- Prepare X: hourly → 15-min interpolated --------
    df_x = df_x.set_index("time").sort_index()
    df_x_15 = df_x.resample("15T").interpolate(method="linear")
    df_x_15.reset_index(inplace=True)
    df_x_15 = df_x_15.rename(columns={"time": "Timestamp"})

    # -------- Limit X and Y also to 2023 cutoff --------
    df_x_15 = df_x_15[(df_x_15["Timestamp"] >= start_date) & (df_x_15["Timestamp"] <= end_date)]
    df_y = df_y[(df_y["Timestamp"] >= start_date) & (df_y["Timestamp"] <= end_date)]

    # -------- Merge GHI (IST) into X (IST) --------
    df_x_15 = pd.merge_asof(
        df_x_15.sort_values("Timestamp"),
        df_ghi.sort_values("Timestamp"),
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("15min")
    )

    # -------- Compute solar zenith & cosine --------
    timestamps_local = df_x_15["Timestamp"].dt.tz_localize(tz_ist, nonexistent="shift_forward")
    solpos = solarposition.get_solarposition(
        timestamps_local,
        latitude=lat,
        longitude=lon
    )
    df_x_15["zenith"] = solpos["zenith"].values
    df_x_15["cos_zenith"] = np.cos(np.radians(df_x_15["zenith"].values))

    # -------- Merge X + Y on Timestamp --------
    df_merge = pd.merge(df_y, df_x_15, on="Timestamp", how="inner")
    df_merge = df_merge.dropna(subset=["GHI"])

    # -------- Chronological Split --------
    n = len(df_merge)
    train_end = int(0.9 * n)
    val_end   = int(0.95 * n)

    df_train = df_merge.iloc[:train_end]
    df_val   = df_merge.iloc[train_end:val_end]
    df_test  = df_merge.iloc[val_end:]


    # -------- Save to disk --------
    site_save = os.path.join(save_dir, site_name)
    os.makedirs(site_save, exist_ok=True)

    df_merge.to_csv(os.path.join(site_save, f"{site_name}_merged_full.csv"), index=False)
    df_train.to_csv(os.path.join(site_save, f"{site_name}_train.csv"), index=False)
    df_val.to_csv(os.path.join(site_save, f"{site_name}_val.csv"), index=False)
    df_test.to_csv(os.path.join(site_save, f"{site_name}_test.csv"), index=False)

    # -------- Summary --------
    print(f"✅ {site_name} processed successfully!")
    print(f"   Total samples : {len(df_merge)}")
    print(f"   Train/Val/Test: {len(df_train)} / {len(df_val)} / {len(df_test)}")
    print(f"   Date range    : {df_merge['Timestamp'].iloc[0]} → {df_merge['Timestamp'].iloc[-1]}")


# ========= MAIN LOOP =========
for site, (lat, lon) in site_coords.items():
    process_site(site, lat, lon)

print("\n🎯 All sites processed and saved successfully!")
