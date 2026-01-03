# =========================================================
# PLOTTING UTILITIES for EnhancedPowerNet
# (Multi-Axis Forecast Plot + Scatter Fit Plot)
# =========================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_column(df, candidates):
    """Return the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the columns found: {candidates}")


# ---------- 1️⃣ Multi-Axis Forecast Plot (Fixed Date Range) ----------
def plot_fixed_day_forecast(
    df_test,
    site_name,
    base_dir,
    start_date,
    end_date,
):
    """
    Multi-axis power forecast plot for a fixed date range.
    Overlays Actual, Predicted, and GHI curves.
    """

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "axes.linewidth": 1.2,
    })

    # ---------- Resolve column names ----------
    ts_col = _resolve_column(df_test, ["timestamp", "Timestamp", "time"])
    actual_col = _resolve_column(df_test, ["Actual_Power_MW", "Actual_Power", "Actual", "actual"])
    pred_col = _resolve_column(df_test, ["Predicted_Power_MW", "Predicted_Power"])
    ghi_col = _resolve_column(df_test, ["ghi", "GHI"])

    df_test[ts_col] = pd.to_datetime(df_test[ts_col], errors="coerce")

    mask = (
        (df_test[ts_col].dt.date >= pd.to_datetime(start_date).date()) &
        (df_test[ts_col].dt.date <= pd.to_datetime(end_date).date())
    )
    df_plot = df_test.loc[mask]

    if df_plot.empty:
        print(f"⚠️ No data found for {start_date} → {end_date}")
        return None

    colors = {
        "actual": "#0072B2",
        "predicted": "#000000",
        "ghi": "#009E73",
    }

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ---------- Power ----------
    ax1.plot(
        df_plot[ts_col],
        df_plot[actual_col],
        lw=2.2,
        color=colors["actual"],
        label="Actual Power",
    )

    ax1.plot(
        df_plot[ts_col],
        df_plot[pred_col],
        lw=2.0,
        linestyle="--",
        color=colors["predicted"],
        label="Predicted Power",
    )

    ax1.set_ylabel("Power (MW)")
    ax1.set_title(f"{site_name}: Actual vs Predicted Power ({start_date} → {end_date})")
    ax1.grid(alpha=0.3)

    # ---------- GHI ----------
    ax2 = ax1.twinx()
    ax2.plot(
        df_plot[ts_col],
        df_plot[ghi_col],
        lw=2.2,
        alpha=0.6,
        color=colors["ghi"],
        label="GHI (W/m²)",
    )
    ax2.set_ylabel("GHI (W/m²)", color=colors["ghi"])
    ax2.tick_params(axis="y", colors=colors["ghi"])

    # ---------- Legend ----------
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", frameon=False)

    plt.tight_layout()

    out_path = os.path.join(
        base_dir,
        f"multi_axis_forecast_{site_name}_{start_date}_{end_date}.png"
    )
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"📈 Multi-axis forecast plot saved → {out_path}")
    return out_path


# ---------- 2️⃣ Scatter Fit Plot ----------
def plot_scatter_fit(df_test, site_name, base_dir, r2, mae, rmse):

    actual_col = _resolve_column(df_test, ["Actual_Power_MW", "Actual_Power", "Actual", "actual"])
    pred_col = _resolve_column(df_test, ["Predicted_Power_MW", "Predicted_Power"])

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df_test[actual_col],
        df_test[pred_col],
        alpha=0.5,
        edgecolor="k",
        linewidth=0.4,
        color="#E69F00",
    )

    lims = [0, max(df_test[actual_col].max(), df_test[pred_col].max())]
    plt.plot(lims, lims, "k--", lw=1)

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Power (MW)")
    plt.ylabel("Predicted Power (MW)")
    plt.title(
        f"{site_name} — Model Fit\n"
        f"R²={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}"
    )
    plt.grid(alpha=0.4)

    out_path = os.path.join(base_dir, f"scatter_fit_{site_name}.png")
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()

    print(f"📊 Scatter plot saved → {out_path}")
    return out_path
