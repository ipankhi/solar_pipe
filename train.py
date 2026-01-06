# ================================================
# MAIN SCRIPT: Teacher → Student Training Pipeline
# ================================================
from train.train import train_powernet
from config.config_mosdac import (
    SITE_NAME,
    STUDENT_FEATURES,
    TEACHER_FEATURES
)

if __name__ == "__main__":

    print("\n🚀 STEP 1: Training TEACHER")
    teacher_results = train_powernet(
        site_name=SITE_NAME,
        mode="teacher",
        feature_cols_teacher=TEACHER_FEATURES
    )

    print("\n🚀 STEP 2: Training STUDENT (no distillation yet)")
    student_results = train_powernet(
        site_name=SITE_NAME,
        mode="student",
        feature_cols_student=STUDENT_FEATURES,
        feature_cols_teacher=TEACHER_FEATURES
    )

    print("\n✅ Training completed!")
    print(f"Teacher best val loss : {teacher_results['best_val_loss']:.6f}")
    print(f"Student best val loss : {student_results['best_val_loss']:.6f}")
