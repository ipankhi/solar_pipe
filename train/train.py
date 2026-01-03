# =========================================================
# TRAIN FUNCTION: Physics-Guided EnhancedPowerNet_v4
# =========================================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config.config_mosdac import *
from data.data_utils import (
    load_power,
    add_features,
    preprocess_data,
    scale_features
)
from models.power_net import PowerNet
from models.loss_functions import AsymSmoothL1_Physics


def train_powernet(
    site_name: str = SITE_NAME,
    save_dir: str = SAVE_DIR,
    train_path: str = TRAIN_PATH,
    val_path: str = VAL_PATH,
    test_path: str = TEST_PATH,
    feature_cols: list = FEATURE_COLS,
    hyperparams: dict = HYPERPARAMS,
):
    """
    Train the Physics-Guided PowerNet
    - GHI augmentation ONLY during training
    - Climatology-based baseline
    - Multi-GPU via DataParallel
    """

    # =====================================================
    # 0. SETUP
    # =====================================================
    os.makedirs(save_dir, exist_ok=True)
    print(f"🚀 Training PowerNet for {site_name}")
    print(f"🖥️  Device: {DEVICE}")

    rng = np.random.default_rng(seed=42)

    # =====================================================
    # 1. LOAD DATA
    # =====================================================
    print("📥 Loading data ...")
    df_train, df_val, df_test = map(load_power, [train_path, val_path, test_path])

    # =====================================================
    # 2. FEATURE ENGINEERING
    # =====================================================
    print("⚙️ Adding engineered features ...")
    df_train = add_features(df_train, mode="train", rng=rng)
    df_val   = add_features(df_val,   mode="infer")
    df_test  = add_features(df_test,  mode="infer")

    # =====================================================
    # 3. TARGET PREPROCESSING
    # =====================================================
    print("📊 Computing targets ...")
    df_train, df_val, df_test = preprocess_data(
        df_train, df_val, df_test, save_dir
    )

    # =====================================================
    # 4. FEATURE SCALING
    # =====================================================
    print("📏 Scaling features ...")
    x_train, x_val, _ = scale_features(
        df_train, df_val, df_test, feature_cols, save_dir
    )

    y_train = df_train["y_rel"].values.reshape(-1, 1)
    y_val   = df_val["y_rel"].values.reshape(-1, 1)

    # =====================================================
    # 5. TORCH TENSORS
    # =====================================================
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
    x_val_t   = torch.tensor(x_val,   dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).to(DEVICE)

    # Physics loss MUST use true GHI
    ghi_train = torch.tensor(df_train["ghi"].values, dtype=torch.float32).to(DEVICE)
    ghi_val   = torch.tensor(df_val["ghi"].values,   dtype=torch.float32).to(DEVICE)

    # =====================================================
    # 6. MODEL + MULTI-GPU
    # =====================================================
    print("🧠 Initializing model ...")

    model = PowerNet(
        input_dim=len(feature_cols),
        hidden=hyperparams["hidden"],
        drop=hyperparams["dropout"]
    )

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧮 Parameters:")
    print(f"   • Total     : {total_params:,}")
    print(f"   • Trainable : {trainable_params:,}")

    criterion = AsymSmoothL1_Physics()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparams["scheduler_T0"],
        eta_min=hyperparams["scheduler_eta_min"]
    )

    # =====================================================
    # 7. TRAINING LOOP
    # =====================================================
    print(f"📈 Training for {hyperparams['epochs']} epochs ...")

    best_val = float("inf")
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(hyperparams["epochs"]):

        # ---------------- TRAIN ----------------
        model.train()
        optimizer.zero_grad()

        pred_train = model(x_train_t)
        loss_train = criterion(pred_train, y_train_t, ghi_train)

        loss_train.backward()
        optimizer.step()

        # ---------------- VALIDATION ----------------
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            loss_val = criterion(pred_val, y_val_t, ghi_val)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        scheduler.step()

        # ---------------- CHECKPOINT ----------------
        if loss_val < best_val:
            best_val = loss_val
            wait = 0

            # IMPORTANT: save underlying model if DataParallel
            state_dict = (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict()
            )

            torch.save(
                state_dict,
                os.path.join(save_dir, "best_powernet.pt")
            )
        else:
            wait += 1
            if wait > hyperparams["patience"]:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        # ---------------- LOG ----------------
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:04d} | "
                f"Train={loss_train.item():.6f} | "
                f"Val={loss_val.item():.6f}"
            )

    print("✅ Training complete!")

    # =====================================================
    # 8. LOSS PLOT
    # =====================================================
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Physics-Guided Loss")
    plt.title(f"PowerNet ({site_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300)
    plt.close()

    print(f"📊 Model & plots saved to {save_dir}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "save_dir": save_dir,
    }
