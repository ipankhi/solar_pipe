# =========================================================
# CONFIG FILE: MOSDAC Physics-Guided EnhancedPowerNet Setup
# =========================================================

import os
import torch

# === Site Information ===
SITE_NAME = "Chandrapur_33kV"

# === Directory Paths ===
BASE_DIR = "/mnt/DATA1/pankhi/forecast/solar_pipe/DATA"
SAVE_PATH = "/mnt/DATA1/pankhi/forecast/solar_pipe/results"
SPLIT_DIR = f"{BASE_DIR}/split/Maharashtra"
SAVE_DIR  = f"{SAVE_PATH}/Maharashtra/{SITE_NAME}_powernet_with_temp"

os.makedirs(SAVE_DIR, exist_ok=True)

# === File Paths ===
TRAIN_PATH = f"{SPLIT_DIR}/train/{SITE_NAME}.csv"
VAL_PATH   = f"{SPLIT_DIR}/val/{SITE_NAME}.csv"
# TEST_PATH  = f"{SPLIT_DIR}/{SITE_NAME}_test.csv"
TEST_PATH = f"{SPLIT_DIR}/test/{SITE_NAME}.csv"

# === Feature List ===
TEACHER_FEATURES = [
    "csi",
    "ghi",
    "cos_zenith",
    "ghi_sq",
    "ghi_cu",
    "t2m"
]
STUDENT_FEATURES = [
    "csi",
    "ghi",
    "cos_zenith",
    "ghi_sq",
    "ghi_cu"
]



# === Hyperparameters ===
HYPERPARAMS = {
    "hidden": 256,
    "dropout": 0.15,          # slightly reduced (climatology stabilizes learning)
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 2500,
    "patience": 250,
    "scheduler_T0": 100,
    "scheduler_eta_min": 1e-5
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"