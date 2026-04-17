"""
Network Attack Analysis System — CICIDS2017
Analyzes network traffic data to detect attacks and generate defense strategies.

Usage:
    python main.py                  # Train on CICIDS2017 and analyze
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_cicids, get_dataset_info
from src.preprocessing import Preprocessor
from src.models import train_model, get_feature_importance

MODEL_PATH = Path("output/model.joblib")
PREPROCESSOR_PATH = Path("output/preprocessor.joblib")
META_PATH = Path("output/meta.joblib")
from src.evaluation import (
    RESULTS_DIR,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_attack_distribution,
)
from src.strategies import generate_strategies


def _load_and_split():
    """Load CICIDS2017 and split into train/test DataFrames."""
    print("\n[STEP 1] Loading CICIDS2017 dataset...")
    df = load_cicids(sample_frac=0.1)  # 10% sample for faster iteration
    info = get_dataset_info(df)
    print(f"  {info['total_records']} records, {info['features']} features, "
          f"{info['attack_types']} attack types")
    print(f"  Categories: {info['attack_categories']}")

    print("\n[STEP 2] Splitting into train/test (80/20)...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["attack_category"]
    )
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    return df_train, df_test


def _load_saved_model():
    """Try to load a previously saved model, preprocessor, and metadata."""
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and META_PATH.exists():
        print("\n[STEP 3] Loading saved model (skipping preprocessing & training)...")
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        meta = joblib.load(META_PATH)
        print(f"[+] Loaded model from {MODEL_PATH}")
        return model, meta["model_name"], preprocessor, meta["feature_cols"], meta["target_names"]
    return None


def _train_system(df_train):
    """Preprocess training data and train the model."""
    print("\n[STEP 3] Preprocessing training data...")
    preprocessor = Preprocessor()
    X_train, y_train, feature_cols = preprocessor.fit_transform(df_train)
    target_names = preprocessor.get_target_names()

    print("\n[STEP 4] Training XGBoost classifier...")
    model, model_name = train_model(X_train, y_train)

    # Save model, preprocessor, and metadata for future runs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump({"model_name": model_name, "feature_cols": feature_cols, "target_names": target_names}, META_PATH)
    print(f"[+] Model saved to {MODEL_PATH}")

    return model, model_name, preprocessor, feature_cols, target_names


def _analyze_data(model, model_name, preprocessor, feature_cols, target_names, df, label: str):
    """Run attack analysis on a DataFrame and output results."""
    X, y_true = preprocessor.transform(df)
    y_pred = model.predict(X)

    # --- Attack distribution ---
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    attack_count = sum(c for u, c in zip(unique, counts) if target_names[u] != "benign")

    print(f"\n{'=' * 60}")
    print(f"  ATTACK ANALYSIS — {label}")
    print(f"{'=' * 60}")
    print(f"  Total records:    {total}")
    print(f"  Attacks detected: {attack_count} ({attack_count/total*100:.1f}%)")
    print(f"  Normal traffic:   {total - attack_count} ({(total-attack_count)/total*100:.1f}%)")
    print(f"\n  {'Category':<12} {'Count':>8} {'Percent':>8}")
    print(f"  {'─' * 30}")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {target_names[u]:<12} {c:>8} {c/total*100:>7.1f}%")

    # --- Evaluation ---
    print(f"\n[*] Evaluating accuracy against known labels...")
    metrics = evaluate_model(model, X, y_true, model_name, target_names)
    plot_confusion_matrix(
        metrics["confusion_matrix"], target_names, model_name,
        save_path=RESULTS_DIR / f"confusion_matrix.png",
    )

    # --- Feature importance ---
    imp = get_feature_importance(model, feature_cols)
    if imp:
        plot_feature_importance(
            imp, model_name, top_n=15,
            save_path=RESULTS_DIR / "feature_importance.png",
        )

    # --- Attack distribution plot ---
    plot_attack_distribution(
        y_pred, target_names,
        title=f"Detected Attack Distribution — {label}",
        save_path=RESULTS_DIR / "attack_distribution.png",
    )

    # --- Defense strategies ---
    print(f"\n[*] Generating defense strategies...")
    results = [{"model": model_name, "f1_macro": metrics["f1_macro"],
                "accuracy": metrics["accuracy"], "y_pred": y_pred}]
    generate_strategies(y_true, y_pred, target_names, results)

    # --- Save predictions (sample to avoid huge output file) ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "predictions.csv"
    import pandas as pd
    pred_df = pd.DataFrame({
        "true_category": [target_names[t] for t in y_true],
        "predicted_category": [target_names[p] for p in y_pred],
    })
    pred_df.to_csv(output_path, index=False)
    print(f"[+] Predictions saved to: {output_path}")

    return y_pred


def main():
    print("=" * 60)
    print("  Network Attack Analysis System — CICIDS2017")
    print("=" * 60)

    df_train, df_test = _load_and_split()

    saved = _load_saved_model()
    if saved:
        model, model_name, preprocessor, feature_cols, target_names = saved
    else:
        model, model_name, preprocessor, feature_cols, target_names = _train_system(df_train)

    print("\n[STEP 5] Analyzing test data...")
    _analyze_data(model, model_name, preprocessor, feature_cols, target_names,
                  df_test, "CICIDS2017 Test Set")

    print("\n[DONE] All results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
