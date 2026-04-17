"""ML model for network attack classification."""

import numpy as np
from xgboost import XGBClassifier


def build_model() -> XGBClassifier:
    """Return the optimized attack classifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple:
    """Train XGBoost classifier on SMOTE-balanced data.

    Returns:
        (model, model_name)
    """
    model = build_model()
    print("[*] Training XGBoost classifier...")
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"    Train accuracy: {train_acc:.4f}")
    return model, "xgboost"


def get_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from the model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: float(imp) for name, imp in pairs}
    return {}
