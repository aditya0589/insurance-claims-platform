"""
Candidate Machine Learning Models for Insurance Fraud Detection.

Constructs self-contained scikit-learn Pipelines pairing preprocessing
with candidate estimators (interpretable baseline, ensemble, and tuned ensemble).
"""

from typing import Any, Dict
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ml.config import RANDOM_STATE


def build_logistic_regression_pipeline(
    preprocessor: ColumnTransformer,
    max_iter: int = 1000,
    c_param: float = 1.0,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Constructs an interpretable linear baseline pipeline.
    """
    classifier = LogisticRegression(
        C=c_param,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 300,
    max_depth: Any = None,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Constructs an ensemble Random Forest pipeline with balanced class weights.
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_tuned_random_forest_pipeline(
    preprocessor: ColumnTransformer,
    n_estimators: int = 400,
    max_depth: int = 12,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Constructs a regularized/tuned Random Forest pipeline designed to reduce
    variance and improve fraud recall stability.
    """
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_candidate_models(preprocessor: ColumnTransformer) -> Dict[str, Dict[str, Any]]:
    """
    Factory function returning candidate model pipelines and metadata.
    """
    return {
        "logistic_regression_baseline": {
            "pipeline": build_logistic_regression_pipeline(preprocessor),
            "model_type": "LogisticRegression",
            "description": "Interpretable linear baseline with balanced class weights",
            "params": {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "random_forest_baseline": {
            "pipeline": build_random_forest_pipeline(preprocessor),
            "model_type": "RandomForestClassifier",
            "description": "Ensemble Random Forest (300 trees, balanced weights)",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 300,
                "max_depth": "None",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
        "random_forest_tuned": {
            "pipeline": build_tuned_random_forest_pipeline(preprocessor),
            "model_type": "RandomForestClassifier",
            "description": "Tuned Random Forest with tree depth and leaf regularization",
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 400,
                "max_depth": 12,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        },
    }
