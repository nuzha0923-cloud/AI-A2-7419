"""
Classification modeling module for Musanze Harvest Intelligence Service.
Compares baseline with at least two classifiers.
Reports precision, recall, F1, confusion matrix, and ROC-AUC.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_CLASSIFICATION,
    preprocess_pipeline,
    fit_preprocessor,
)
from src.validation import VERIFICATION_CODE, LLLL


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, Any]:
    """Compute classification metrics."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    
    return metrics


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    """Train majority class baseline classifier."""
    baseline = DummyClassifier(strategy="most_frequent", random_state=LLLL)
    baseline.fit(X_train, y_train)
    return baseline


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train Logistic Regression classifier."""
    model = LogisticRegression(
        random_state=LLLL,
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200) -> RandomForestClassifier:
    """Train Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=LLLL,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200) -> GradientBoostingClassifier:
    """Train Gradient Boosting Classifier."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        random_state=LLLL,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a classifier on test data."""
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    return compute_classification_metrics(y_test, y_pred, y_prob)


def run_classification_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete classification experiment:
    1. Preprocess data
    2. Train baseline and candidate classifiers
    3. Evaluate on test set
    4. Select best model based on F1 score
    5. Save results and model
    """
    # Preprocess
    artifacts = preprocess_pipeline(df)
    X_train = artifacts["X_train_transformed"]
    X_test = artifacts["X_test_transformed"]
    y_clf_train = artifacts["y_clf_train"].values
    y_clf_test = artifacts["y_clf_test"].values
    preprocessor = artifacts["preprocessor"]
    
    # Train models
    models = {
        "baseline_majority": train_baseline(X_train, y_clf_train),
        "logistic_regression": train_logistic_regression(X_train, y_clf_train),
        "random_forest": train_random_forest(X_train, y_clf_train, n_estimators=200),
        "gradient_boosting": train_gradient_boosting(X_train, y_clf_train, n_estimators=200),
    }
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        metrics = evaluate_classifier(model, X_test, y_clf_test)
        results[name] = metrics
        print(f"{name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
              f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']}")
    
    # Select best model (excluding baseline) based on F1
    candidate_models = {k: v for k, v in results.items() if k != "baseline_majority"}
    best_model_name = max(candidate_models, key=lambda k: candidate_models[k]["f1"])
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]
    
    # Compare with baseline
    baseline_metrics = results["baseline_majority"]
    
    # Selection reason
    f1_improvement = best_metrics["f1"] - baseline_metrics["f1"]
    reason = (
        f"Selected {best_model_name} over baseline. "
        f"F1 improved by {f1_improvement:.4f} "
        f"(baseline F1={baseline_metrics['f1']:.4f} vs {best_model_name} F1={best_metrics['f1']:.4f}). "
        f"Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, "
        f"ROC-AUC={best_metrics['roc_auc']}."
    )
    
    # Save best model with preprocessor in a pipeline
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])
    
    # Save artifacts
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "classification_pipeline.joblib")
    
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    classification_results = {
        "verification_code": VERIFICATION_CODE,
        "seed": LLLL,
        "baseline": {"model": "DummyClassifier(strategy='most_frequent')", **baseline_metrics},
        "candidates": {name: {"model": type(model).__name__, **metrics} for name, model in models.items() if name != "baseline_majority"},
        "selected_model": best_model_name,
        "selected_model_type": type(best_model).__name__,
        "selected_metrics": best_metrics,
        "selection_reason": reason,
    }
    
    with open(artifacts_dir / "classification_metrics.json", "w") as f:
        json.dump(classification_results, f, indent=2)
    
    print(f"\n[OK] Best classification model: {best_model_name}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {best_metrics['roc_auc']}")
    print(f"  Confusion Matrix: {best_metrics['confusion_matrix']}")
    print(f"  Reason: {reason}")
    
    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_pipeline": final_pipeline,
        "preprocessor": preprocessor,
        "X_test": artifacts["X_test"],  # Original DataFrame for pipeline
        "X_test_transformed": X_test,  # Transformed array for direct model use
        "y_clf_test": y_clf_test,
        "feature_names": artifacts["feature_names"],
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Create confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Low Risk (0)", "High Risk (1)"],
                yticklabels=["Low Risk (0)", "High Risk (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - High Risk Classification")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix plot to {output_path}")


if __name__ == "__main__":
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
    df = pd.read_csv(csv_path)
    results = run_classification_experiment(df)
    
    # Plot confusion matrix
    y_pred = results["best_pipeline"].predict(results["X_test"])
    plot_confusion_matrix(results["y_clf_test"], y_pred, Path(__file__).parent.parent / "artifacts" / "plots" / "confusion_matrix.png")