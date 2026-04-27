# ─────────────────────────────────────────────
#  ml/train.py
#  Trains XGBoost on dataset.csv and saves model.pkl
#  Run: python ml/train.py
# ─────────────────────────────────────────────

import sys
import os
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATASET_PATH, MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN,
    WEIGHT_COLUMN, SENIORITY_MAP, DOMAIN_MAP, XGB_PARAMS, TEST_SIZE, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(path: str) -> tuple:
    """
    Loads dataset.csv, encodes categoricals, returns X, y, weights.
    All encoding uses the fixed maps from config.py — no fit_transform.
    """
    logger.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path, keep_default_na=False)
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Encode categorical columns using fixed maps
    df["seniority_fit_encoded"] = df["seniority_fit"].map(SENIORITY_MAP).fillna(1).astype(int)
    df["resume_domain_encoded"] = df["resume_domain"].map(DOMAIN_MAP).fillna(0).astype(int)
    df["job_domain_encoded"]    = df["job_domain"].map(DOMAIN_MAP).fillna(0).astype(int)

    # Verify all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    weights = df[WEIGHT_COLUMN].astype(float)

    logger.info(f"Features: {X.shape[1]} columns")
    logger.info(f"Label balance — 0: {(y==0).sum():,}  1: {(y==1).sum():,}")

    return X, y, weights


def train():
    # ── Load ───────────────────────────────────
    X, y, weights = prepare_dataset(DATASET_PATH)

    # ── Split ──────────────────────────────────
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # ── Imbalance handling ─────────────────────
    zeros = (y_train == 0).sum()
    ones  = (y_train == 1).sum()
    ratio = round(zeros / ones, 2)
    logger.info(f"scale_pos_weight = {ratio}")

    # ── Train ──────────────────────────────────
    params = {**XGB_PARAMS, "scale_pos_weight": ratio}
    model = XGBClassifier(**params)

    logger.info("Training XGBoost...")
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    logger.info("Training complete.")

    # ── Evaluate ───────────────────────────────
    y_pred  = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    y_prob  = model.predict_proba(X_test)[:, 1]

    logger.info("\n" + "="*50)
    logger.info("EVALUATION ON TEST SET")
    logger.info("="*50)
    logger.info(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"  True Negatives  (correct rejections): {cm[0][0]:,}")
    logger.info(f"  False Positives (wrongly selected):   {cm[0][1]:,}")
    logger.info(f"  False Negatives (missed candidates):  {cm[1][0]:,}")
    logger.info(f"  True Positives  (correct selections): {cm[1][1]:,}")

    # ── Feature importance ─────────────────────
    logger.info("\nFeature Importance:")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    for feat, score in importance.sort_values(ascending=False).items():
        logger.info(f"  {feat:<30} {score:.4f}")

    # ── Save ───────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
