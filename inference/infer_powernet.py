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
from data.data_utils import load_power, add_features
from models.power_net import PowerNet
from utils.plot_utils import plot_fixed_day_forecast, plot_scatter_fit

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from torchsummary import summary


def evaluate_powernet(
    site_name: str = SITE_NAME,
    save_dir: str = SAVE_DIR,
    test_path: str = TEST_PATH,
    feature_cols: list = FEATURE_COLS,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Evaluation script for climatology-based EnhancedPowerNet
    Safe for timestamp naming, DataParallel, and plotting
    """

    # =====================================================
    # 0. SETUP
    # =====================================================
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"🚀 Evaluating model on device: {device}")

    # =====================================================
    # 1. LOAD MODEL (ARCHITECTURE MUST MATCH TRAINING)
    # =====================================================
    model_path = os.path.join(save_dir, "best_powernet.pt")

    model = PowerNet(
        input_dim=len(feature_cols),
        hidden=HYPERPARAMS["hidden"],
        drop=HYPERPARAMS["dropout"],
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {model_path}")

    # =====================================================
    # 2. LOAD + FEATURE ENGINEER TEST DATA
    # =====================================================
    print("📥 Loading test data ...")
    df_test = load_power(test_path)
    df_test = add_features(df_test, mode="infer")

    # =====================================================
    # 2.1 NORMALIZE TIMESTAMP COLUMN (CRITICAL FIX)
    # =====================================================
# =====================================================
# NORMALIZE TIMESTAMP COLUMN (FINAL, SIMPLE)
# =====================================================

    # print(df_test)
    df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"])
    df_test = df_test.sort_values("Timestamp").reset_index(drop=True)


    # =====================================================
    # 3. SCALE FEATURES
    # =====================================================
    scaler_x = load(os.path.join(save_dir, "scaler_x.pkl"))["scaler_x"]

    assert len(feature_cols) == scaler_x.n_features_in_, (
        "❌ Feature count mismatch between training and evaluation"
    )

    x_test = scaler_x.transform(df_test[feature_cols])
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)

    # =====================================================
    # 4. MODEL INFERENCE
    # =====================================================
    with torch.no_grad():
        y_pred = model(x_test_t).cpu().numpy().ravel()

    # =====================================================
    # 5. RECOVER ABSOLUTE POWER
    # =====================================================
    print(df_test)
    df_test["Predicted_Power"] = (
        df_test["climatology_power"] * (1.0 + y_pred)
    ).clip(lower=0.0)

    df_test["Actual_Power"] = df_test["Actual"]

    # =====================================================
    # 6. METRICS
    # =====================================================
    y_true = df_test["Actual_Power"].values
    y_hat  = df_test["Predicted_Power"].values

    mae  = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    r2   = r2_score(y_true, y_hat)

    print(f"📊 Metrics | R²={r2:.4f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

    # =====================================================
    # 7. SAVE PREDICTIONS
    # =====================================================
    out_csv = os.path.join(save_dir, f"{site_name}_test_predictions.csv")
    df_test.to_csv(out_csv, index=False)
    print(f"✅ Predictions saved → {out_csv}")

    # =====================================================
    # 8. AUTO DATE SELECTION FOR PLOTTING
    # =====================================================
    if start_date is None or end_date is None:
        unique_days = df_test["Timestamp"].dt.date.unique()
        if len(unique_days) >= 2:
            start_date = unique_days[-2]
            end_date   = unique_days[-1]
        else:
            start_date = unique_days[0]
            end_date   = unique_days[0]

    print(f"📅 Plotting from {start_date} → {end_date}")

    # =====================================================
    # 9. PLOTS
    # =====================================================
    print("📈 Generating plots ...")
    plot_fixed_day_forecast(df_test, site_name, save_dir, start_date, end_date)
    plot_scatter_fit(df_test, site_name, save_dir, r2, mae, rmse)

    print("🎯 Evaluation complete.")
    print(f"🧪 Test samples: {len(df_test)}")

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pred_path": out_csv,
    }
