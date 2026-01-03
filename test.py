# ================================================
# MAIN SCRIPT: Run Training for EnhancedPowerNet_v4
# ================================================
from inference.infer_powernet import evaluate_powernet

if __name__ == "__main__":
    results = evaluate_powernet()
    print("\n🏁 Final Results:")
    print(f"R² = {results['r2']:.4f}, MAE = {results['mae']:.3f}, RMSE = {results['rmse']:.3f}")
    print(f"Predictions saved at: {results['pred_path']}")
