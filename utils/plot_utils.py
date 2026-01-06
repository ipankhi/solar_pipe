# =========================================================
# PLOTTING UTILITIES for EnhancedPowerNet
# (Daily Forecast + Scatter Fit)
# =========================================================
import os
import matplotlib.pyplot as plt
import pandas as pd


def _resolve_column(df, candidates):
    """Return the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns found: {candidates}")


# =========================================================
# 1️⃣ DAILY MULTI-AXIS FORECAST PLOT
# =========================================================
def plot_fixed_day_forecast(
    df,
    site_name,
    save_dir,
    start_date,
    end_date,
):
    """
    Multi-axis power forecast plot for a SINGLE day.
    Overlays Actual, Predicted, and GHI.
    """

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "axes.linewidth": 1.2,
    })

    # ---------- Resolve columns ----------
    ts_col = _resolve_column(df, ["Timestamp", "timestamp", "time"])
    actual_col = _resolve_column(df, ["Actual_Power", "Actual", "actual"])
    pred_col = _resolve_column(df, ["Predicted_Power"])
    ghi_col = _resolve_column(df, ["ghi", "GHI"])

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    mask = (
        (df[ts_col].dt.date >= pd.to_datetime(start_date).date()) &
        (df[ts_col].dt.date <= pd.to_datetime(end_date).date())
    )
    df_plot = df.loc[mask]

    if df_plot.empty:
        print(f"⚠️ No data found for {start_date}")
        return None

    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    # ---------- Power ----------
    ax1.plot(
        df_plot[ts_col],
        df_plot[actual_col],
        lw=2.2,
        label="Actual Power",
    )

    ax1.plot(
        df_plot[ts_col],
        df_plot[pred_col],
        lw=2.0,
        linestyle="--",
        label="Predicted Power",
    )

    ax1.set_ylabel("Power (MW)")
    ax1.set_title(f"{site_name} | {start_date}")
    ax1.grid(alpha=0.3)

    # ---------- GHI ----------
    ax2 = ax1.twinx()
    ax2.plot(
        df_plot[ts_col],
        df_plot[ghi_col],
        lw=2.0,
        alpha=0.5,
        label="GHI (W/m²)",
    )
    ax2.set_ylabel("GHI (W/m²)")

    # ---------- Legend ----------
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    plt.tight_layout()

    out_path = os.path.join(
        save_dir,
        f"forecast_{site_name}_{start_date}.png"
    )
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"📈 Saved daily forecast → {out_path}")
    return out_path


# =========================================================
# 2️⃣ SCATTER FIT PLOT (FULL TEST SET)
# =========================================================
def plot_scatter_fit(df, site_name, save_dir, r2, mae, rmse):

    actual_col = _resolve_column(df, ["Actual_Power", "Actual", "actual"])
    pred_col = _resolve_column(df, ["Predicted_Power"])

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df[actual_col],
        df[pred_col],
        alpha=0.45,
        edgecolor="k",
        linewidth=0.4,
    )

    lims = [0, max(df[actual_col].max(), df[pred_col].max())]
    plt.plot(lims, lims, "k--", lw=1)

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Power (MW)")
    plt.ylabel("Predicted Power (MW)")
    plt.title(
        f"{site_name}\n"
        f"R²={r2:.3f} | MAE={mae:.2f} | RMSE={rmse:.2f}"
    )
    plt.grid(alpha=0.4)

    out_path = os.path.join(save_dir, f"scatter_fit_{site_name}.png")
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()

    print(f"📊 Saved scatter fit → {out_path}")
    return out_path
