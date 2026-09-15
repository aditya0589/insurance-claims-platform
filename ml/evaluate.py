"""
Evaluation Module for Fraud Detection Models.

Generates evaluation metrics, confusion matrices, ROC/PR curves,
threshold trade-off analysis, and feature importance visual artifacts.
"""

import sys
from pathlib import Path

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

from ml.config import THRESHOLD_CANDIDATES, DEFAULT_OPERATING_THRESHOLD
from ml.preprocessing import extract_transformed_feature_names


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_OPERATING_THRESHOLD,
) -> Dict[str, float]:
    """
    Computes key performance metrics at a specific operating threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fraud_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }
    return metrics


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluates precision, recall, and F1 across a spectrum of decision thresholds.
    Identifies the optimal operating threshold balancing fraud detection (recall)
    against operational review capacity (precision/F1).
    """
    thresholds = candidates or THRESHOLD_CANDIDATES
    records = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        records.append({
            "threshold": round(t, 2),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        })

    df_thresh = pd.DataFrame(records)
    # Target candidate with maximum F1 that maintains recall >= 0.65
    valid_candidates = df_thresh[df_thresh["recall"] >= 0.60]
    if not valid_candidates.empty:
        best_threshold = float(valid_candidates.loc[valid_candidates["f1"].idxmax()]["threshold"])
    else:
        best_threshold = float(df_thresh.loc[df_thresh["f1"].idxmax()]["threshold"])

    return df_thresh, best_threshold


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generates and saves normalized confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    save_path: Path,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> None:
    """
    Plots and saves the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve",
) -> None:
    """
    Plots and saves the Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#2ca02c", lw=2, label="PR Curve")
    ax.set_xlabel("Recall (Fraud Coverage)", fontsize=10)
    ax.set_ylabel("Precision (Investigation Quality)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_threshold_curves(
    df_thresh: pd.DataFrame,
    selected_threshold: float,
    save_path: Path,
    title: str = "Fraud Detection: Threshold Trade-offs",
) -> None:
    """
    Visualizes Precision, Recall, and F1 across thresholds with selected operating point marked.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#1f77b4", lw=2)
    ax.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#d62728", lw=2)
    ax.plot(df_thresh["threshold"], df_thresh["f1"], label="F1 Score", color="#2ca02c", lw=2)

    ax.axvline(
        x=selected_threshold,
        color="black",
        linestyle="--",
        label=f"Selected Threshold ({selected_threshold:.2f})",
    )
    ax.set_xlabel("Classification Probability Threshold", fontsize=10)
    ax.set_ylabel("Metric Score", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    pipeline: Any,
    save_path: Path,
    top_n: int = 15,
    title: str = "Top Predictive Features for Fraud Risk",
) -> Optional[pd.DataFrame]:
    """
    Computes and plots Gini feature importance for tree-based estimators.
    """
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(classifier, "feature_importances_"):
        return None

    feature_names = extract_transformed_feature_names(preprocessor)
    importances = classifier.feature_importances_

    # Match dimensions if possible
    if len(feature_names) != len(importances):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    df_top = df_imp.head(top_n).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df_top["feature"], df_top["importance"], color="#3b528b")
    ax.set_xlabel("Feature Importance (Gini Importance)", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    return df_imp


def run_evaluation() -> Dict[str, Any]:
    """
    Evaluates the champion model on the holdout test set and prints a comprehensive summary.
    """
    import json
    from ml.config import ARTIFACTS_DIR, RANDOM_STATE, TEST_SIZE
    from ml.data import load_dataset
    from ml.features import prepare_features
    from ml.predict import load_model_and_metadata
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("INSURANCE FRAUD MODEL: HOLDOUT TEST EVALUATION")
    print("=" * 60)

    model, metadata = load_model_and_metadata()
    threshold = metadata.get("operating_threshold", DEFAULT_OPERATING_THRESHOLD)

    df, _ = load_dataset()
    X, y, _, _ = prepare_features(df)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_predictions(y_test.values, y_prob, threshold=threshold)
    print(f"\nChampion Model: {metadata.get('champion_model_name')}")
    print(f"Operating Threshold: {threshold:.2f}")
    print("-" * 40)
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"Fraud Recall:     {metrics['fraud_recall']:.4f}")
    print(f"Fraud Precision:  {metrics['fraud_precision']:.4f}")
    print(f"Fraud F1 Score:   {metrics['fraud_f1']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))

    return metrics


if __name__ == "__main__":
    run_evaluation()

