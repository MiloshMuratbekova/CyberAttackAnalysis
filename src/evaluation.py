"""Model evaluation: metrics, reports, and visualizations."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output"


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    target_names: list[str],
) -> dict:
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X_test)

    # Use explicit label indices so target_names always aligns
    labels = list(range(len(target_names)))

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0, labels=labels),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0, labels=labels),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0, labels=labels),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, labels=labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "y_pred": y_pred,
    }

    print(f"\n{'=' * 60}")
    print(f"  Model: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    model_name: str,
    save_path: Path | None = None,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[+] Saved: {save_path}")
    plt.close(fig)


def plot_feature_importance(
    importances: dict[str, float],
    model_name: str,
    top_n: int = 15,
    save_path: Path | None = None,
) -> None:
    """Plot top-N feature importances."""
    top = dict(list(importances.items())[:top_n])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(top.keys())[::-1], list(top.values())[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[+] Saved: {save_path}")
    plt.close(fig)


def plot_attack_distribution(
    y: np.ndarray,
    target_names: list[str],
    title: str = "Attack Category Distribution",
    save_path: Path | None = None,
) -> None:
    """Plot distribution of attack categories."""
    unique, counts = np.unique(y, return_counts=True)
    labels = [target_names[i] for i in unique]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(labels))
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, (lbl, cnt) in enumerate(zip(labels, counts)):
        ax.text(i, cnt + max(counts) * 0.01, str(cnt), ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[+] Saved: {save_path}")
    plt.close(fig)
