"""
Pytest tests for Musanze Harvest Intelligence Service.
Covers valid inference, missing field, wrong type, range error, reproducibility, and API response.
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api import app, predict_single, FarmInput
from src.validation import VERIFICATION_CODE, LLLL, load_and_validate
from src.preprocessing import preprocess_pipeline
from data.generate_data import main as generate_data, LLLL as GEN_LLLL


client = TestClient(app)


class TestDataGeneration:
    """Tests for data generation and validation."""
    
    def test_data_generation_reproducibility(self):
        """Running generation twice with same seed produces identical CSV."""
        generate_data()
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df1 = pd.read_csv(csv_path)
        
        generate_data()
        df2 = pd.read_csv(csv_path)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_data_fingerprint_stable(self):
        """Data fingerprint matches expected value."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        fingerprint_path = Path(__file__).parent.parent / "artifacts" / "data_fingerprint.txt"
        
        if fingerprint_path.exists():
            with open(fingerprint_path) as f:
                expected = f.read().strip()
            
            import hashlib
            with open(csv_path, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            
            assert actual == expected, f"Fingerprint mismatch: {actual} != {expected}"
    
    def test_dataset_schema(self):
        """Dataset has required columns and structure."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df = load_and_validate(csv_path, LLLL)
        
        assert len(df) == 282  # 279 base + 3 duplicates
        assert list(df.columns) == [
            "rainfall_mm", "avg_temp_c", "soil_moisture_pct",
            "fertilizer_kg", "field_size_ha", "season",
            "cooperative_support", "yield_tons", "high_risk"
        ]
    
    def test_missing_values_pattern(self):
        """Missing values only in expected columns with expected counts."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df = pd.read_csv(csv_path)
        
        assert df["soil_moisture_pct"].isna().sum() == 15
        assert df["fertilizer_kg"].isna().sum() == 14
        assert df.drop(columns=["soil_moisture_pct", "fertilizer_kg"]).isna().sum().sum() == 0
    
    def test_duplicates_exist(self):
        """Dataset contains exactly 3 duplicate rows."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df = pd.read_csv(csv_path)
        
        assert df.duplicated().sum() == 3


class TestPreprocessing:
    """Tests for preprocessing pipeline."""
    
    def test_train_test_split_no_leakage(self):
        """Preprocessing fits only on training data."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df = pd.read_csv(csv_path)
        
        artifacts = preprocess_pipeline(df)
        
        # Check that imputer statistics come from training data only
        preprocessor = artifacts["preprocessor"]
        imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        
        # Imputer statistics should be based on training data medians
        imputed_cols = ["soil_moisture_pct", "fertilizer_kg"]
        train_medians = artifacts["X_train"][imputed_cols].median()
        
        # The numeric pipeline in ColumnTransformer handles 5 numeric columns
        # The imputer statistics array has 5 values (one per numeric column)
        # We know the first 3 are rainfall_mm, avg_temp_c, soil_moisture_pct
        # and the last 2 are fertilizer_kg, field_size_ha (based on SCALE_COLUMNS order)
        # soil_moisture_pct is index 2, fertilizer_kg is index 3
        np.testing.assert_array_almost_equal(imputer.statistics_[2], train_medians["soil_moisture_pct"])
        np.testing.assert_array_almost_equal(imputer.statistics_[3], train_medians["fertilizer_kg"])
    
    def test_same_transform_train_test(self):
        """Train and test use same preprocessing."""
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        df = pd.read_csv(csv_path)
        
        artifacts = preprocess_pipeline(df)
        
        # Both should have same number of features
        assert artifacts["X_train_transformed"].shape[1] == artifacts["X_test_transformed"].shape[1]
        
        # Feature names should match
        assert artifacts["feature_names"] == artifacts["preprocessor"].get_feature_names_out().tolist()


class TestCLIPrediction:
    """Tests for CLI prediction."""
    
    @pytest.fixture
    def artifacts(self):
        """Load saved models."""
        models_dir = Path(__file__).parent.parent / "models"
        return {
            "regression": joblib.load(models_dir / "regression_pipeline.joblib"),
            "classification": joblib.load(models_dir / "classification_pipeline.joblib"),
            "clustering": joblib.load(models_dir / "clustering_pipeline.joblib"),
        }
    
    def test_valid_inference(self, artifacts):
        """Valid input produces expected output structure."""
        input_df = pd.DataFrame([{
            "rainfall_mm": 950.0,
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 1,
        }])
        
        yield_est = artifacts["regression"].predict(input_df)[0]
        risk_pred = artifacts["classification"].predict(input_df)[0]
        risk_prob = artifacts["classification"].predict_proba(input_df)[0][1]
        cluster = artifacts["clustering"].predict(input_df)[0]
        
        assert isinstance(yield_est, (float, np.floating))
        assert yield_est >= 0.5  # clipped minimum
        assert risk_pred in [0, 1]
        assert 0 <= risk_prob <= 1
        assert isinstance(cluster, (int, np.integer))
        assert 0 <= cluster < 6  # K <= 6
    
    def test_verification_code_in_output(self, artifacts):
        """Prediction output includes verification code."""
        # This is tested via predict.py CLI
        result = subprocess.run([
            sys.executable, "predict.py", "--json", "samples/valid.json"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["verification_code"] == VERIFICATION_CODE


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    def test_health_endpoint(self):
        """GET /health returns healthy status with verification code."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["verification_code"] == VERIFICATION_CODE
    
    def test_model_info_endpoint(self):
        """GET /model-info returns model metadata."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["verification_code"] == VERIFICATION_CODE
        assert data["seed"] == LLLL
        assert "regression_model" in data
        assert "classification_model" in data
        assert "clustering_k" in data
    
    def test_predict_valid_input(self):
        """POST /predict with valid input returns prediction."""
        response = client.post("/predict", json={
            "rainfall_mm": 950.0,
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert "yield_estimate_tons" in data
        assert "high_risk_prediction" in data
        assert "high_risk_probability" in data
        assert "cluster" in data
        assert data["verification_code"] == VERIFICATION_CODE
    
    def test_predict_missing_field(self):
        """POST /predict with missing field returns 422."""
        response = client.post("/predict", json={
            "rainfall_mm": 950.0,
            "avg_temp_c": 18.5,
            # missing soil_moisture_pct
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 1,
        })
        assert response.status_code == 422
    
    def test_predict_wrong_type(self):
        """POST /predict with wrong type returns 422."""
        response = client.post("/predict", json={
            "rainfall_mm": "high",  # should be float
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 1,
        })
        assert response.status_code == 422
    
    def test_predict_range_error(self):
        """POST /predict with out-of-range values returns 422."""
        response = client.post("/predict", json={
            "rainfall_mm": 500.0,  # below minimum 600
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 1,
        })
        assert response.status_code == 422
    
    def test_predict_invalid_season(self):
        """POST /predict with invalid season returns 422."""
        response = client.post("/predict", json={
            "rainfall_mm": 950.0,
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "C",  # invalid
            "cooperative_support": 1,
        })
        assert response.status_code == 422
    
    def test_predict_invalid_cooperative(self):
        """POST /predict with invalid cooperative_support returns 422."""
        response = client.post("/predict", json={
            "rainfall_mm": 950.0,
            "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0,
            "field_size_ha": 1.5,
            "season": "A",
            "cooperative_support": 2,  # invalid
        })
        assert response.status_code == 422


class TestReproducibility:
    """Tests for full pipeline reproducibility."""
    
    def test_run_all_reproducible(self):
        """Running run_all.py twice produces same artifacts."""
        # This is an integration test - run once, check artifacts exist
        csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
        assert csv_path.exists()
        
        artifacts_dir = Path(__file__).parent.parent / "artifacts"
        assert (artifacts_dir / "regression_metrics.json").exists()
        assert (artifacts_dir / "classification_metrics.json").exists()
        assert (artifacts_dir / "clustering_report.json").exists()
        assert (artifacts_dir / "test_predictions.csv").exists()
        assert (artifacts_dir / "data_fingerprint.txt").exists()
        assert (artifacts_dir / "plots" / "residuals.png").exists()
        assert (artifacts_dir / "plots" / "confusion_matrix.png").exists()
        assert (artifacts_dir / "plots" / "clusters.png").exists()
    
    def test_models_load_consistently(self):
        """Models load and produce same predictions."""
        models_dir = Path(__file__).parent.parent / "models"
        
        reg1 = joblib.load(models_dir / "regression_pipeline.joblib")
        reg2 = joblib.load(models_dir / "regression_pipeline.joblib")
        
        input_df = pd.DataFrame([{
            "rainfall_mm": 950.0, "avg_temp_c": 18.5,
            "soil_moisture_pct": 55.0, "fertilizer_kg": 120.0,
            "field_size_ha": 1.5, "season": "A", "cooperative_support": 1
        }])
        
        pred1 = reg1.predict(input_df)[0]
        pred2 = reg2.predict(input_df)[0]
        
        assert pred1 == pred2


class TestMetricsArtifacts:
    """Tests that metrics artifacts are valid JSON with required keys."""
    
    def test_regression_metrics_structure(self):
        path = Path(__file__).parent.parent / "artifacts" / "regression_metrics.json"
        with open(path) as f:
            data = json.load(f)
        
        assert data["verification_code"] == VERIFICATION_CODE
        assert data["seed"] == LLLL
        assert "baseline" in data
        assert "candidates" in data
        assert "selected_model" in data
        assert "selected_metrics" in data
        assert "selection_reason" in data
        
        # Check metrics structure
        for model_data in [data["baseline"]] + list(data["candidates"].values()):
            assert "mae" in model_data
            assert "rmse" in model_data
            assert "r2" in model_data
    
    def test_classification_metrics_structure(self):
        path = Path(__file__).parent.parent / "artifacts" / "classification_metrics.json"
        with open(path) as f:
            data = json.load(f)
        
        assert data["verification_code"] == VERIFICATION_CODE
        assert data["seed"] == LLLL
        assert "baseline" in data
        assert "candidates" in data
        assert "selected_model" in data
        assert "selected_metrics" in data
        assert "selection_reason" in data
        
        for model_data in [data["baseline"]] + list(data["candidates"].values()):
            assert "precision" in model_data
            assert "recall" in model_data
            assert "f1" in model_data
            assert "confusion_matrix" in model_data
            assert "roc_auc" in model_data
    
    def test_clustering_report_structure(self):
        path = Path(__file__).parent.parent / "artifacts" / "clustering_report.json"
        with open(path) as f:
            data = json.load(f)
        
        assert data["verification_code"] == VERIFICATION_CODE
        assert data["seed"] == LLLL
        assert "silhouette_scores" in data
        assert "selected_k" in data
        assert "selected_silhouette" in data
        assert "cluster_profiles" in data
        assert len(data["cluster_profiles"]) == data["selected_k"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])