# =========================================================
# EVALUATION + DAILY PLOTTING (Prediction Setup)
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


def evaluate_powernet(
    site_name: str = SITE_NAME,
    save_dir: str = SAVE_DIR,
    test_path: str = TEST_PATH,
    feature_cols: list = STUDENT_FEATURES,
):
    """
    Evaluation script for PowerNet (prediction mode)
    - Predicts normalized power (cf)
    - Converts to absolute power
    - Generates one plot per day
    """

    # =====================================================
    # 0. SETUP
    # =====================================================
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"🚀 Evaluating PowerNet on device: {device}")

    # =====================================================
    # 1. LOAD MODEL
    # =====================================================
    model_path = os.path.join(save_dir, "best_student_powernet.pt")

    model = PowerNet(
        input_dim=len(feature_cols),
        hidden=HYPERPARAMS["hidden"],
        drop=HYPERPARAMS["dropout"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✅ Model loaded from: {model_path}")

    # =====================================================
    # 2. LOAD + FEATURE ENGINEER TEST DATA
    # =====================================================
    print("📥 Loading test data ...")
    df_test = load_power(test_path)
    df_test = add_features(df_test)

    # =====================================================
    # 3. CLEAN & SORT
    # =====================================================
    df_test["Timestamp"] = pd.to_datetime(df_test["Timestamp"])
    df_test = df_test.sort_values("Timestamp").reset_index(drop=True)

    # Drop invalid rows
    cols_needed = feature_cols + ["cf", "available_capacity"]
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.dropna(subset=cols_needed)

    print(f"🧪 Test samples after cleaning: {len(df_test)}")

    # =====================================================
    # 4. SCALE FEATURES
    # =====================================================
    scaler_x = load(os.path.join(save_dir, "scaler_x_student.pkl"))

    assert scaler_x.n_features_in_ == len(feature_cols), (
        "❌ Feature count mismatch between training and evaluation"
    )

    x_test = scaler_x.transform(df_test[feature_cols])
    x_test_t = torch.tensor(x_test, dtype=torch.float32).to(device)

    # =====================================================
    # 5. MODEL INFERENCE
    # =====================================================
    with torch.no_grad():
        cf_pred = model(x_test_t).cpu().numpy().ravel()

    # Safety clamp
    cf_pred = np.clip(cf_pred, 0.0, 1.2)

    # =====================================================
    # 6. ABSOLUTE POWER
    # =====================================================
    df_test["Predicted_Power"] = np.clip(
        cf_pred * df_test["available_capacity"].values,
        a_min=0.0,
        a_max=None
    )

    df_test["Actual_Power"] = df_test["actual"]

    # =====================================================
    # 7. METRICS
    # =====================================================
    y_true = df_test["Actual_Power"].values
    y_hat  = df_test["Predicted_Power"].values

    mae  = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    r2   = r2_score(y_true, y_hat)

    print(f"📊 Metrics | R²={r2:.4f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

    # =====================================================
    # 8. SAVE PREDICTIONS
    # =====================================================
    out_csv = os.path.join(save_dir, f"{site_name}_test_predictions.csv")
    df_test.to_csv(out_csv, index=False)
    print(f"✅ Predictions saved → {out_csv}")

    # =====================================================
    # 9. DAILY PLOTS
    # =====================================================
    print("📈 Generating daily plots ...")

    df_test["date"] = df_test["Timestamp"].dt.date

    for day in sorted(df_test["date"].unique()):
        plot_fixed_day_forecast(
            df=df_test,
            site_name=site_name,
            save_dir=save_dir,
            start_date=day,
            end_date=day,
        )

    # =====================================================
    # 10. SCATTER FIT
    # =====================================================
    plot_scatter_fit(
        df=df_test,
        site_name=site_name,
        save_dir=save_dir,
        r2=r2,
        mae=mae,
        rmse=rmse
    )

    print("🎯 Evaluation complete.")

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pred_path": out_csv,
    }


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    evaluate_powernet()
