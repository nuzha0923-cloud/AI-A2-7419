"""
Preprocessing module for Musanze Harvest Intelligence Service.
Handles train-only fitting, imputation, scaling, encoding, and train/test split.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.validation import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    BINARY_COLUMNS,
    LLLL,
    VERIFICATION_CODE,
)

# Feature columns (excluding targets)
FEATURE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
]

TARGET_REGRESSION = "yield_tons"
TARGET_CLASSIFICATION = "high_risk"

# Columns that need imputation (have missing values by design)
IMPUTE_COLUMNS = ["soil_moisture_pct", "fertilizer_kg"]

# Numeric features that need scaling
SCALE_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
]


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - Imputes missing values in soil_moisture_pct and fertilizer_kg with median
    - Scales numeric features
    - One-hot encodes season
    - Passes through cooperative_support (already binary)
    """
    # Numeric pipeline: impute + scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # Categorical pipeline: one-hot encode
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    
    # Binary columns pass through as-is
    binary_pipeline = "passthrough"
    
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, SCALE_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ("binary", binary_pipeline, BINARY_COLUMNS[:-1]),  # exclude high_risk (target)
    ], remainder="drop", verbose_feature_names_out=False)
    
    return preprocessor


def prepare_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into features, regression target, and classification target."""
    X = df[FEATURE_COLUMNS].copy()
    y_reg = df[TARGET_REGRESSION].copy()
    y_clf = df[TARGET_CLASSIFICATION].copy()
    return X, y_reg, y_clf


def split_data(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    test_size: float = 0.2,
    random_state: int = LLLL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/test sets with stratification on classification target.
    Uses the same split for both regression and classification.
    """
    # We stratify on the classification target to maintain class balance
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessing pipeline on training data only."""
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train)
    return preprocessor


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transform data using fitted preprocessor."""
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get feature names after preprocessing."""
    return preprocessor.get_feature_names_out().tolist()


def save_preprocessor(preprocessor: ColumnTransformer, path: Path):
    """Save fitted preprocessor to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load fitted preprocessor from disk."""
    return joblib.load(path)


def preprocess_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = LLLL,
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline:
    1. Split data
    2. Fit preprocessor on training data only
    3. Transform both train and test
    Returns dictionary with all artifacts.
    """
    X, y_reg, y_clf = prepare_features_targets(df)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(
        X, y_reg, y_clf, test_size, random_state
    )
    
    preprocessor = fit_preprocessor(X_train)
    X_train_transformed = transform_data(preprocessor, X_train)
    X_test_transformed = transform_data(preprocessor, X_test)
    feature_names = get_feature_names(preprocessor)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train,
        "y_clf_test": y_clf_test,
        "X_train_transformed": X_train_transformed,
        "X_test_transformed": X_test_transformed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    # Test preprocessing
    from data.generate_data import main as gen_data
    import pandas as pd
    
    csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
    df = pd.read_csv(csv_path)
    artifacts = preprocess_pipeline(df)
    print(f"Train shape: {artifacts['X_train_transformed'].shape}")
    print(f"Test shape: {artifacts['X_test_transformed'].shape}")
    print(f"Features: {artifacts['feature_names']}")