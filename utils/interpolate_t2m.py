import pandas as pd
import xarray as xr
from glob import glob
from pathlib import Path

# ===================== INPUTS =====================
T2M_GLOB = "/mnt/DATA1/pankhi/forecast/ERA5/downloaded/era5_t2m/*.nc"
SITES_CSV = "/mnt/DATA1/pankhi/forecast/DATA/MH_sites.csv"

OUT_DIR = Path("/mnt/DATA1/pankhi/forecast/DATA/era5_by_pos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INTERP_METHOD = "linear"
CHUNK_TIME = 240
# =================================================


# -------------------- Read site locations --------------------
sites_df = pd.read_csv(SITES_CSV)
sites_df.columns = [c.strip().lower() for c in sites_df.columns]

sites_df = sites_df.rename(columns={
    "posname": "pos_name",
    "latitude": "lat",
    "longitude": "lon"
})

assert {"pos_name", "lat", "lon"}.issubset(sites_df.columns)


# -------------------- Open ERA5 t2m --------------------
paths = sorted(glob(T2M_GLOB))
ds = xr.open_mfdataset(paths, combine="by_coords")

# standardize names
if "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})
if "lat" in ds.coords:
    ds = ds.rename({"lat": "latitude"})
if "lon" in ds.coords:
    ds = ds.rename({"lon": "longitude"})
if "expver" in ds.dims:
    ds = ds.isel(expver=0)

# rename data variable → t2m
ds = ds.rename({list(ds.data_vars)[0]: "t2m"})


# -------------------- Longitude fix --------------------
if float(ds.longitude.min()) >= 0 and (sites_df["lon"] < 0).any():
    sites_df["lon"] = (sites_df["lon"] + 360) % 360


# -------------------- Chunk --------------------
ds = ds.chunk({"time": CHUNK_TIME})


# -------------------- Interpolate --------------------
interp = ds.interp(
    latitude=xr.DataArray(
        sites_df["lat"].values,
        dims="site",
        coords={"site": sites_df["pos_name"].values}
    ),
    longitude=xr.DataArray(
        sites_df["lon"].values,
        dims="site",
        coords={"site": sites_df["pos_name"].values}
    ),
    method=INTERP_METHOD,
)


# -------------------- Save per pos_name --------------------
for pos in interp.site.values:
    df = (
        interp.sel(site=pos)
        .to_dataframe()
        .reset_index()
        [["time", "t2m"]]
    )

    # OPTIONAL: Kelvin → Celsius
    # df["t2m"] = df["t2m"] - 273.15

    out_file = OUT_DIR / f"{pos}_ERA5_t2m_hourly.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved → {out_file}")
