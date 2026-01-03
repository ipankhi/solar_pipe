# =========================================================
# EVALUATION + PLOTTING for EnhancedPowerNet_v4
# =========================================================
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import load

from config.config_mosdac import *
from data.data_utils import load_power, clean_and_engineer, add_features
from models.power_net import PowerNet
from utils.plot_utils import plot_fixed_day_forecast, plot_scatter_fit  # ✅ import your plotting utilities


def evaluate_powernet(
    site_name: str = SITE_NAME,
    save_dir: str = SAVE_DIR,
    test_path: str = TEST_PATH,
    feature_cols: list = FEATURE_COLS,
    start_date: str = "2023-12-21",
    end_date: str = "2023-12-22",
):
    """
    Evaluate the trained EnhancedPowerNet model on the test set
    and generate forecast and scatter-fit plots.
    """

    # === 0. Setup ===
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"🚀 Evaluating model on device: {device}")

    # === 1. Load Model ===
    model_path = os.path.join(save_dir, "best_ghi_correction.pt")
    model = PowerNet(len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded from: {model_path}")

    # === 2. Load and Preprocess Test Data ===
    df_test = load_power(test_path)
    df_test = clean_and_engineer(df_test)
    df_test = add_features(df_test)

    # --- Load normalization stats ---
    P_mean, P_std, GHI_mean, GHI_std, bias_mean, bias_std = np.load(
        os.path.join(save_dir, "power_ghi_mean_std.npy")
    )
    df_test["forecast_bias_z"] = (df_test["forecast_bias"] - bias_mean) / (bias_std + 1e-6)
    df_test["ghi_norm"] = (df_test["ghi"] - GHI_mean) / (GHI_std + 1e-6)

    # --- Scale features ---
    scaler_x = load(os.path.join(save_dir, "scaler_x.pkl"))["scaler_x"]
    x_test = scaler_x.transform(df_test[feature_cols])
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)

    # === 3. Model Inference ===
    with torch.no_grad():
        y_pred_rel = model(x_test_t).cpu().numpy().ravel()

    # --- Recover actual power values ---
    power_norm_pred = df_test["ghi_norm"] * (1 + y_pred_rel)
    df_test["Predicted_Power_MW"] = (power_norm_pred * P_std) + P_mean
    df_test["Actual_Power_MW"] = df_test["Actual"]

    # === 4. Compute Metrics ===
    y_true = df_test["Actual_Power_MW"].values
    y_pred = df_test["Predicted_Power_MW"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 Metrics | R²={r2:.4f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

    # === 5. Save Results ===
    results_csv = os.path.join(save_dir, f"{site_name}_test_predictions.csv")
    df_test.to_csv(results_csv, index=False)
    print(f"✅ Predictions saved → {results_csv}")

    # === 6. Generate Plots ===
    print("📈 Generating forecast and scatter plots ...")
    plot_fixed_day_forecast(df_test, site_name, save_dir, start_date, end_date)
    plot_scatter_fit(df_test, site_name, save_dir, r2, mae, rmse)

    print("🎯 Evaluation complete.")
    return {"r2": r2, "mae": mae, "rmse": rmse, "pred_path": results_csv}



