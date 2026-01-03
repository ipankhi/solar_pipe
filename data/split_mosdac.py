import pandas as pd
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
SITE_CSV = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH_sites.csv")
DATA_DIR = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/MH")
OUT_ROOT = Path("/mnt/DATA1/pankhi/forecast/solar_pipe/DATA/split")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# READ SITE META
# =========================================================
sites_df = pd.read_csv(SITE_CSV)
sites_df.columns = [c.strip().lower() for c in sites_df.columns]

assert {"pos_name", "state"}.issubset(sites_df.columns)

# =========================================================
# HELPERS
# =========================================================
def normalize_underscore(name: str) -> str:
    return "_".join(str(name).strip().split())

# =========================================================
# MAIN LOOP
# =========================================================
for _, row in sites_df.iterrows():

    pos   = row["pos_name"]
    state = row["state"]
    site  = normalize_underscore(pos)

    site_file = DATA_DIR / f"{site}.csv"

    if not site_file.exists():
        print(f"❌ Missing site file: {site_file}")
        continue

    print(f"\n🔹 Processing {site} ({state})")

    # ---------------- READ ----------------
    df = pd.read_csv(site_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        print("  ❌ Empty file, skipping")
        continue

    # ---------------- DATE RANGE ----------------
    start_time = df["timestamp"].iloc[0]
    end_time   = df["timestamp"].iloc[-1]

    print(f"   📅 {start_time.date()} → {end_time.date()}")

    # ---------------- SPLIT DATES ----------------
    test_start = end_time.normalize() - pd.Timedelta(days=6)
    val_start  = end_time.normalize() - pd.Timedelta(days=13)

    # ---------------- SPLIT DATA ----------------
    df_train = df[df["timestamp"] < val_start]
    df_val   = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)]
    df_test  = df[df["timestamp"] >= test_start]

    if df_train.empty or df_val.empty or df_test.empty:
        print("  ⚠️ One split empty, skipping")
        continue

    # ---------------- OUTPUT DIRS ----------------
    train_dir = OUT_ROOT / state / "train"
    val_dir   = OUT_ROOT / state / "val"
    test_dir  = OUT_ROOT / state / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- SAVE ----------------
    df_train.to_csv(train_dir / f"{site}.csv", index=False)
    df_val.to_csv(val_dir / f"{site}.csv", index=False)
    df_test.to_csv(test_dir / f"{site}.csv", index=False)

    # ---------------- SUMMARY ----------------
    print(f"   ✅ Train rows: {len(df_train)}")
    print(f"   ✅ Val rows  : {len(df_val)}")
    print(f"   ✅ Test rows : {len(df_test)}")

print("\n🎯 ALL SITES SPLIT (TRAIN / VAL / TEST) SUCCESSFULLY")
