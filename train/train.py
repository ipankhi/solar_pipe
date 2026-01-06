# =========================================================
# TRAIN FUNCTION: PowerNet with Teacher–Student Distillation
# =========================================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config.config_mosdac import *
from data.data_utils import load_power, add_features, scale_features
from models.power_net import PowerNet
from models.loss_functions import DistillSmoothL1


def train_powernet(
    site_name: str = SITE_NAME,
    save_dir: str = SAVE_DIR,
    train_path: str = TRAIN_PATH,
    val_path: str = VAL_PATH,
    feature_cols_student: list = STUDENT_FEATURES,
    feature_cols_teacher: list | None = None,
    hyperparams: dict = HYPERPARAMS,
    mode: str = "student"   # "teacher" | "student"
):
    """
    PowerNet training
    - Teacher: uses temperature (T2M)
    - Student: distilled from teacher
    """

    assert mode in {"teacher", "student"}
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"\n🚀 Training PowerNet ({mode.upper()}) for {site_name}")
    print(f"🖥️  Device: {device}")

    # =====================================================
    # 1. LOAD DATA
    # =====================================================
    df_train = load_power(train_path)
    df_val   = load_power(val_path)

    df_train = add_features(df_train)
    df_val   = add_features(df_val)

    # =====================================================
    # 2. FEATURE SELECTION
    # =====================================================
    if mode == "teacher":
        assert feature_cols_teacher is not None
        feature_cols = feature_cols_teacher
    else:
        feature_cols = feature_cols_student

    # =====================================================
    # 3. CLEAN DATA
    # =====================================================
    cols_needed = feature_cols + ["cf"]
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)
    df_val   = df_val.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)

    assert len(df_train) > 100, "❌ Train set too small"
    assert len(df_val) > 100, "❌ Val set too small"

    # =====================================================
    # 4. TARGET
    # =====================================================
    y_train = torch.tensor(df_train["cf"].values, dtype=torch.float32).view(-1, 1).to(device)
    y_val   = torch.tensor(df_val["cf"].values,   dtype=torch.float32).view(-1, 1).to(device)

    # =====================================================
    # 5. FEATURE SCALING
    # =====================================================
    x_train, x_val, _ = scale_features(mode,
        df_train, df_val, df_val, feature_cols, save_dir
    )

    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_val_t   = torch.tensor(x_val,   dtype=torch.float32).to(device)

    # =====================================================
    # 6. MODEL
    # =====================================================
    student = PowerNet(
        input_dim=len(feature_cols),
        hidden=hyperparams["hidden"],
        drop=hyperparams["dropout"]
    ).to(device)

    print(f"🧮 Student params: {sum(p.numel() for p in student.parameters()):,}")

    # =====================================================
    # 7. LOAD TEACHER (STUDENT MODE ONLY)
    # =====================================================
    teacher = None
    if mode == "student":
        assert feature_cols_teacher is not None

        teacher = PowerNet(
            input_dim=len(feature_cols_teacher),
            hidden=hyperparams["hidden"],
            drop=hyperparams["dropout"]
        ).to(device)

        teacher_ckpt = os.path.join(save_dir, "best_teacher_powernet.pt")
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        teacher.eval()

        for p in teacher.parameters():
            p.requires_grad = False

        # Teacher inputs
        x_train_teacher, x_val_teacher, _ = scale_features(
            mode=="teacher",df_train, df_val, df_val, feature_cols_teacher, save_dir
        )

        x_train_teacher_t = torch.tensor(x_train_teacher, dtype=torch.float32).to(device)
        x_val_teacher_t   = torch.tensor(x_val_teacher,   dtype=torch.float32).to(device)

        print("🎓 Teacher loaded and frozen")

    # =====================================================
    # 8. LOSS & OPTIMIZER
    # =====================================================
    criterion = DistillSmoothL1(
        alpha=1.0 if mode == "teacher" else hyperparams.get("alpha", 0.6),
        beta=0.1
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=hyperparams.get("lr", 3e-4),
        weight_decay=hyperparams["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparams["scheduler_T0"],
        eta_min=hyperparams["scheduler_eta_min"]
    )

    # =====================================================
    # 9. TRAINING LOOP
    # =====================================================
    best_val = float("inf")
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(hyperparams["epochs"]):

        # ---------- TRAIN ----------
        student.train()
        optimizer.zero_grad()

        pred_s = torch.clamp(student(x_train_t), -0.2, 1.5)

        if mode == "student":
            with torch.no_grad():
                pred_t = torch.clamp(teacher(x_train_teacher_t), -0.2, 1.5)

            loss_train = criterion(pred_student=pred_s, target=y_train, pred_teacher=pred_t)
        else:
            loss_train = criterion(pred_student=pred_s, target=y_train)

        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        optimizer.step()

        # ---------- VALIDATION ----------
        student.eval()
        with torch.no_grad():
            pred_sv = torch.clamp(student(x_val_t), -0.2, 1.5)

            if mode == "student":
                pred_tv = torch.clamp(teacher(x_val_teacher_t), -0.2, 1.5)
                loss_val = criterion(pred_student=pred_sv, target=y_val, pred_teacher=pred_tv)
            else:
                loss_val = criterion(pred_student=pred_sv, target=y_val)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        scheduler.step()

        # ---------- CHECKPOINT ----------
        if loss_val < best_val:
            best_val = loss_val
            wait = 0
            torch.save(
                student.state_dict(),
                os.path.join(save_dir, f"best_{mode}_powernet.pt")
            )
        else:
            wait += 1
            if wait > hyperparams["patience"]:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch+1:04d} | "
                f"Train={loss_train.item():.5f} | "
                f"Val={loss_val.item():.5f}"
            )

    print("✅ Training complete")

    # =====================================================
    # 10. LOSS PLOT
    # =====================================================
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"PowerNet ({mode}) – {site_name}")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_{mode}.png"), dpi=300)
    plt.close()

    return {
        "best_val_loss": best_val,
        "save_dir": save_dir
    }
