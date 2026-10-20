"""
Regression modeling module for Musanze Harvest Intelligence Service.
Compares baseline with at least two regression models.
Reports MAE, RMSE, R² on held-out test set.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_REGRESSION,
    preprocess_pipeline,
    create_preprocessing_pipeline,
    fit_preprocessor,
    transform_data,
)
from src.validation import VERIFICATION_CODE, LLLL


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R² metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> DummyRegressor:
    """Train mean baseline regressor."""
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    return baseline


def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Train Ridge Regression model."""
    model = Ridge(alpha=alpha, random_state=LLLL)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100) -> RandomForestRegressor:
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=LLLL,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100) -> GradientBoostingRegressor:
    """Train Gradient Boosting Regressor."""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        random_state=LLLL,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate a regression model on test data."""
    y_pred = model.predict(X_test)
    return compute_regression_metrics(y_test, y_pred)


def run_regression_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete regression experiment:
    1. Preprocess data
    2. Train baseline and candidate models
    3. Evaluate on test set
    4. Select best model based on RMSE
    5. Save results and model
    """
    # Preprocess
    artifacts = preprocess_pipeline(df)
    X_train = artifacts["X_train_transformed"]
    X_test = artifacts["X_test_transformed"]
    y_reg_train = artifacts["y_reg_train"].values
    y_reg_test = artifacts["y_reg_test"].values
    preprocessor = artifacts["preprocessor"]
    
    # Train models
    models = {
        "baseline_mean": train_baseline(X_train, y_reg_train),
        "linear_regression": train_linear_regression(X_train, y_reg_train),
        "ridge_regression": train_ridge_regression(X_train, y_reg_train, alpha=1.0),
        "random_forest": train_random_forest(X_train, y_reg_train, n_estimators=200),
        "gradient_boosting": train_gradient_boosting(X_train, y_reg_train, n_estimators=200),
    }
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_reg_test)
        results[name] = metrics
        print(f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    # Select best model (excluding baseline) based on RMSE
    candidate_models = {k: v for k, v in results.items() if k != "baseline_mean"}
    best_model_name = min(candidate_models, key=lambda k: candidate_models[k]["rmse"])
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]
    
    # Compare with baseline
    baseline_metrics = results["baseline_mean"]
    
    # Selection reason
    improvement = ((baseline_metrics["rmse"] - best_metrics["rmse"]) / baseline_metrics["rmse"]) * 100
    reason = (
        f"Selected {best_model_name} over baseline. "
        f"RMSE improved by {improvement:.1f}% "
        f"(baseline RMSE={baseline_metrics['rmse']:.4f} vs {best_model_name} RMSE={best_metrics['rmse']:.4f}). "
        f"R² improved from {baseline_metrics['r2']:.4f} to {best_metrics['r2']:.4f}."
    )
    
    # Save best model with preprocessor in a pipeline
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])
    
    # Save artifacts
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "regression_pipeline.joblib")
    
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    regression_results = {
        "verification_code": VERIFICATION_CODE,
        "seed": LLLL,
        "baseline": {"model": "DummyRegressor(strategy='mean')", **baseline_metrics},
        "candidates": {name: {"model": type(model).__name__, **metrics} for name, model in models.items() if name != "baseline_mean"},
        "selected_model": best_model_name,
        "selected_model_type": type(best_model).__name__,
        "selected_metrics": best_metrics,
        "selection_reason": reason,
    }
    
    with open(artifacts_dir / "regression_metrics.json", "w") as f:
        json.dump(regression_results, f, indent=2)
    
    print(f"\n[OK] Best regression model: {best_model_name}")
    print(f"  MAE: {best_metrics['mae']:.4f}")
    print(f"  RMSE: {best_metrics['rmse']:.4f}")
    print(f"  R²: {best_metrics['r2']:.4f}")
    print(f"  Reason: {reason}")
    
    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_pipeline": final_pipeline,
        "preprocessor": preprocessor,
        "X_test": artifacts["X_test"],  # Original DataFrame for pipeline
        "X_test_transformed": X_test,  # Transformed array for direct model use
        "y_reg_test": y_reg_test,
        "feature_names": artifacts["feature_names"],
    }


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Create residuals plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Yield (tons)")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved residuals plot to {output_path}")


if __name__ == "__main__":
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
    df = pd.read_csv(csv_path)
    results = run_regression_experiment(df)
    
    # Plot residuals
    y_pred = results["best_pipeline"].predict(results["X_test"])
    plot_residuals(results["y_reg_test"], y_pred, Path(__file__).parent.parent / "artifacts" / "plots" / "residuals.png")