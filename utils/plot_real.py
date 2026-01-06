import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "/mnt/DATA1/pankhi/forecast/solar_pipe/results_realtime/Chandrapur_33kV_020126.csv"
SITE_NAME = "Fermi Solar 132kV"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# =========================================================
# PLOT (TWIN AXIS)
# =========================================================
fig, ax1 = plt.subplots(figsize=(12, 5))

# ----- GHI (left axis) -----
ax1.plot(
    df["timestamp"],
    df["ghi"],
    color="tab:orange",
    linewidth=2,
    label="GHI (W/m²)"
)
ax1.set_ylabel("GHI (W/m²)", color="tab:orange")
ax1.tick_params(axis="y", labelcolor="tab:orange")

# ----- Prediction (right axis) -----
ax2 = ax1.twinx()
ax2.plot(
    df["timestamp"],
    df["prediction"],
    color="tab:blue",
    linewidth=2,
    label="Predicted Power (MW)"
)
ax2.set_ylabel("Predicted Power (MW)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

# =========================================================
# TITLE & GRID
# =========================================================
ax1.set_title(f"{SITE_NAME} | GHI vs Predicted Power")
ax1.set_xlabel("Time")
ax1.grid(True, linestyle="--", alpha=0.4)

# =========================================================
# LEGEND (combined)
# =========================================================
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.tight_layout()
plt.savefig("/mnt/DATA1/pankhi/forecast/solar_pipe/results_realtime/Chandrapur_33kV_020126.png")
