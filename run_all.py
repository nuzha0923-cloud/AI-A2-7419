#!/usr/bin/env python
"""
Main entry point for Musanze Harvest Intelligence Service.
Recreates data, cleans it, trains all tasks and regenerates every artifact.
Usage: python run_all.py --student-id <email> --seed <LLLL>
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from data.generate_data import main as generate_data, LLLL, VERIFICATION_CODE
from src.validation import load_and_validate
from src.regression import run_regression_experiment, plot_residuals
from src.classification import run_classification_experiment, plot_confusion_matrix
from src.clustering import run_clustering_experiment, plot_clusters
from src.preprocessing import preprocess_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete ML pipeline for Musanze Harvest Intelligence Service")
    parser.add_argument("--student-id", required=True, help="Student email (e.g., ug2424671@ines.ac.rw)")
    parser.add_argument("--seed", type=int, required=True, help="Seed (last 4 digits of registration number)")
    return parser.parse_args()


def validate_student_id(student_id: str, seed: int) -> bool:
    """Validate student ID format and seed match."""
    # Check seed matches expected LLLL
    if seed != LLLL:
        print(f"ERROR: Seed mismatch. Expected {LLLL}, got {seed}")
        return False
    
    # Basic email format check
    if "@" not in student_id or "." not in student_id:
        print(f"ERROR: Invalid student ID format: {student_id}")
        return False
    
    print(f"[OK] Student ID validated: {student_id}")
    print(f"[OK] Seed validated: {seed} (LLLL)")
    print(f"[OK] Verification code: {VERIFICATION_CODE}")
    return True


def create_test_predictions_csv(
    reg_results: dict,
    clf_results: dict,
    clust_results: dict,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create test_predictions.csv with all predictions for test set."""
    # Get test indices from preprocessing artifacts (use the same split as training)
    artifacts = preprocess_pipeline(df)
    test_indices = artifacts["X_test"].index.tolist()
    
    # Get predictions using original DataFrame (not transformed array)
    X_test_df = artifacts["X_test"]
    y_reg_test = artifacts["y_reg_test"]
    y_clf_test = artifacts["y_clf_test"]
    
    reg_pred = reg_results["best_pipeline"].predict(X_test_df)
    clf_pred = clf_results["best_pipeline"].predict(X_test_df)
    clf_prob = clf_results["best_pipeline"].predict_proba(X_test_df)[:, 1]
    cluster_labels = clust_results["labels"][test_indices]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "row_id": test_indices,
        "actual_yield_tons": y_reg_test.values,
        "predicted_yield_tons": reg_pred,
        "yield_error": y_reg_test.values - reg_pred,
        "actual_high_risk": y_clf_test.values,
        "predicted_high_risk": clf_pred,
        "high_risk_probability": clf_prob,
        "cluster": cluster_labels,
    })
    
    return results_df


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MUSANZE HARVEST INTELLIGENCE SERVICE - FULL PIPELINE")
    print("=" * 60)
    print(f"Student ID: {args.student_id}")
    print(f"Seed: {args.seed}")
    print(f"Verification Code: {VERIFICATION_CODE}")
    print()
    
    # Validate inputs
    if not validate_student_id(args.student_id, args.seed):
        sys.exit(1)
    
    # Step 1: Generate data
    print("\n[1/7] Generating dataset...")
    generate_data()
    
    # Step 2: Load and validate
    print("\n[2/7] Validating dataset...")
    csv_path = Path(__file__).parent / "data" / "AI_A2_2527419.csv"
    df = load_and_validate(csv_path, args.seed)
    
    # Step 3: Regression
    print("\n[3/7] Training regression models...")
    reg_results = run_regression_experiment(df)
    
    # Step 4: Classification
    print("\n[4/7] Training classification models...")
    clf_results = run_classification_experiment(df)
    
    # Step 5: Clustering
    print("\n[5/7] Running clustering...")
    clust_results = run_clustering_experiment(df)
    
    # Step 6: Generate plots
    print("\n[6/7] Generating plots...")
    plot_residuals(
        reg_results["y_reg_test"],
        reg_results["best_pipeline"].predict(reg_results["X_test"]),
        Path(__file__).parent / "artifacts" / "plots" / "residuals.png"
    )
    plot_confusion_matrix(
        clf_results["y_clf_test"],
        clf_results["best_pipeline"].predict(clf_results["X_test"]),
        Path(__file__).parent / "artifacts" / "plots" / "confusion_matrix.png"
    )
    plot_clusters(
        clust_results["X_scaled"],
        clust_results["labels"],
        Path(__file__).parent / "artifacts" / "plots" / "clusters.png"
    )
    
    # Step 7: Create test_predictions.csv
    print("\n[7/7] Creating test_predictions.csv...")
    test_preds = create_test_predictions_csv(reg_results, clf_results, clust_results, df)
    artifacts_dir = Path(__file__).parent / "artifacts"
    test_preds.to_csv(artifacts_dir / "test_predictions.csv", index=False, float_format="%.4f")
    print(f"[OK] Saved test predictions to {artifacts_dir / 'test_predictions.csv'}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Verification Code: {VERIFICATION_CODE}")
    print(f"Dataset fingerprint: {Path(__file__).parent / 'artifacts' / 'data_fingerprint.txt'}")
    print("\nGenerated artifacts:")
    for artifact in artifacts_dir.rglob("*"):
        if artifact.is_file():
            print(f"  {artifact.relative_to(Path(__file__).parent)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())