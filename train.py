# ================================================
# MAIN SCRIPT: Run Training for EnhancedPowerNet_v4
# ================================================
from train.train import train_powernet

if __name__ == "__main__":
    results = train_powernet()
    print("\n✅ Training summary:")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f"Results saved in: {results['save_dir']}")
