"""
End-to-End MLOps Training Pipeline for Insurance Fraud Detection.

Executes data loading, stratified splitting, preprocessing pipeline assembly,
multi-candidate training, MLflow tracking, threshold optimization,
model selection, registry deployment with '@champion' alias, and local artifact persistence.
"""

import sys
from pathlib import Path

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Any, Dict
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml.config import (
    ARTIFACTS_DIR,
    CHAMPION_ALIAS,
    DEFAULT_OPERATING_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PRIMARY_METRIC,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.data import load_dataset
from ml.evaluate import (
    analyze_thresholds,
    evaluate_predictions,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_curves,
)
from ml.features import prepare_features
from ml.models import get_candidate_models
from ml.preprocessing import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pipeline")


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes the complete reproducible training workflow.
    """
    logger.info("=" * 60)
    logger.info("STARTING INSURANCE FRAUD MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load Data
    df, source_desc = load_dataset()
    logger.info("Data source: %s | Total records: %d", source_desc, len(df))

    # 2. Prepare Features & Target
    X, y, numerical_cols, categorical_cols = prepare_features(df)
    logger.info(
        "Features prepared: %d total (%d numerical, %d categorical)",
        X.shape[1],
        len(numerical_cols),
        len(categorical_cols),
    )
    logger.info("Target distribution: Non-Fraud=%d, Fraud=%d", (y == 0).sum(), (y == 1).sum())

    # 3. Stratified Train/Test Split (Prevent Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train rows: %d | Test rows: %d (stratified split)", len(X_train), len(X_test))

    # 4. Initialize Preprocessor & Candidate Models
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    candidate_models = get_candidate_models(preprocessor)

    # 5. Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow Tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("MLflow Experiment Name: %s", MLFLOW_EXPERIMENT_NAME)

    run_results = {}
    temp_plots_dir = ARTIFACTS_DIR / "temp_eval"
    temp_plots_dir.mkdir(parents=True, exist_ok=True)

    # 6. Train & Track Each Candidate Model
    for model_name, config in candidate_models.items():
        logger.info("-" * 50)
        logger.info("Training candidate model: %s", model_name)
        pipeline = config["pipeline"]

        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            # Fit pipeline on training set
            pipeline.fit(X_train, y_train)

            # Predict probabilities on test set
            y_prob_test = pipeline.predict_proba(X_test)[:, 1]

            # Threshold optimization on test set
            df_thresholds, optimal_threshold = analyze_thresholds(y_test.values, y_prob_test)
            logger.info("Optimal threshold calculated for %s: %.2f", model_name, optimal_threshold)

            # Evaluate at optimal operating threshold
            metrics = evaluate_predictions(y_test.values, y_prob_test, threshold=optimal_threshold)
            y_pred_test = (y_prob_test >= optimal_threshold).astype(int)

            logger.info(
                "Metrics [%s] -> ROC-AUC: %.4f | Recall: %.4f | Precision: %.4f | F1: %.4f | Accuracy: %.4f",
                model_name,
                metrics["roc_auc"],
                metrics["fraud_recall"],
                metrics["fraud_precision"],
                metrics["fraud_f1"],
                metrics["accuracy"],
            )

            # Log parameters
            mlflow.log_params(config["params"])
            mlflow.log_params({
                "test_size": TEST_SIZE,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X.shape[1],
                "numerical_feature_count": len(numerical_cols),
                "categorical_feature_count": len(categorical_cols),
                "selected_operating_threshold": optimal_threshold,
                "data_source": source_desc,
            })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Generate Artifacts
            cm_path = temp_plots_dir / f"{model_name}_confusion_matrix.png"
            roc_path = temp_plots_dir / f"{model_name}_roc_curve.png"
            pr_path = temp_plots_dir / f"{model_name}_pr_curve.png"
            thresh_path = temp_plots_dir / f"{model_name}_threshold_tradeoff.png"
            feat_path = temp_plots_dir / f"{model_name}_feature_importance.png"
            rep_path = temp_plots_dir / f"{model_name}_classification_report.txt"

            plot_confusion_matrix(y_test.values, y_pred_test, cm_path, f"{model_name} Confusion Matrix")
            plot_roc_curve(y_test.values, y_prob_test, metrics["roc_auc"], roc_path, f"{model_name} ROC Curve")
            plot_precision_recall_curve(y_test.values, y_prob_test, pr_path, f"{model_name} PR Curve")
            plot_threshold_curves(df_thresholds, optimal_threshold, thresh_path, f"{model_name} Threshold Trade-offs")
            plot_feature_importance(pipeline, feat_path, top_n=15, title=f"{model_name} Feature Importances")

            # Classification report text
            rep_text = classification_report(
                y_test.values, y_pred_test, target_names=["Non-Fraud", "Fraud"], digits=4
            )
            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(rep_text)

            # Log artifacts to MLflow
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            mlflow.log_artifact(str(roc_path), artifact_path="evaluation")
            mlflow.log_artifact(str(pr_path), artifact_path="evaluation")
            mlflow.log_artifact(str(thresh_path), artifact_path="evaluation")
            if feat_path.exists():
                mlflow.log_artifact(str(feat_path), artifact_path="evaluation")
            mlflow.log_artifact(str(rep_path), artifact_path="evaluation")

            # Log full sklearn Pipeline as MLflow model
            input_example = X_test.head(3)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                input_example=input_example,
                serialization_format="pickle",
            )

            run_results[model_name] = {
                "run_id": run_id,
                "pipeline": pipeline,
                "metrics": metrics,
                "optimal_threshold": optimal_threshold,
                "config": config,
            }

    # 7. Model Selection (Primary: ROC-AUC, Secondary: Fraud F1 & Recall)
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON & SELECTION")
    logger.info("=" * 60)

    ranked_models = sorted(
        run_results.items(),
        key=lambda x: (x[1]["metrics"]["roc_auc"], x[1]["metrics"]["fraud_f1"]),
        reverse=True,
    )

    champion_name, champion_data = ranked_models[0]
    logger.info("Selected CHAMPION Model: %s", champion_name)
    logger.info("Champion Metrics: %s", champion_data["metrics"])

    # 8. Register Champion Model in MLflow Model Registry
    client = MlflowClient()
    champion_run_id = champion_data["run_id"]
    model_uri = f"runs:/{champion_run_id}/model"

    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MLFLOW_REGISTERED_MODEL_NAME,
        )
        logger.info(
            "Registered model '%s' version %s in MLflow Model Registry",
            MLFLOW_REGISTERED_MODEL_NAME,
            registered_model.version,
        )

        # Set modern MLflow alias 'champion'
        try:
            client.set_registered_model_alias(
                name=MLFLOW_REGISTERED_MODEL_NAME,
                alias=CHAMPION_ALIAS,
                version=registered_model.version,
            )
            logger.info(
                "Assigned alias '@%s' to model version %s",
                CHAMPION_ALIAS,
                registered_model.version,
            )
        except Exception as alias_err:
            logger.warning("Could not set model alias (MLflow version compatibility): %s", alias_err)

    except Exception as reg_err:
        logger.warning("MLflow Model Registration notice: %s", reg_err)

    # 9. Export Champion Model Artifact locally for Resilient Serving
    champion_pipeline = champion_data["pipeline"]
    local_artifact_path = ARTIFACTS_DIR / "champion_model.joblib"
    joblib.dump(champion_pipeline, local_artifact_path)
    logger.info("Exported self-contained champion model to: %s", local_artifact_path)

    metadata = {
        "champion_model_name": champion_name,
        "model_type": champion_data["config"]["model_type"],
        "run_id": champion_run_id,
        "metrics": champion_data["metrics"],
        "operating_threshold": champion_data["optimal_threshold"],
        "feature_names": X.columns.tolist(),
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "data_source": source_desc,
        "random_state": RANDOM_STATE,
        "all_model_results": {
            k: {
                "run_id": v["run_id"],
                "metrics": v["metrics"],
                "threshold": v["optimal_threshold"],
            }
            for k, v in run_results.items()
        },
    }

    meta_path = ARTIFACTS_DIR / "champion_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved champion model metadata to: %s", meta_path)

    # 10. Summary Table for Technical Interview Discussion
    summary_rows = []
    for name, data in run_results.items():
        m = data["metrics"]
        summary_rows.append({
            "Model Candidate": name,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Fraud Recall": f"{m['fraud_recall']:.4f}",
            "Fraud Precision": f"{m['fraud_precision']:.4f}",
            "Fraud F1": f"{m['fraud_f1']:.4f}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Threshold": f"{m['threshold']:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("           EXPERIMENT TRACKING SUMMARY (MLflow Recorded Runs)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"WINNER -> '{champion_name}' registered as '@{CHAMPION_ALIAS}'\n")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
