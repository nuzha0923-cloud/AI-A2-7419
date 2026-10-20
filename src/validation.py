"""
Data validation module for Musanze Harvest Intelligence Service.
Validates dataset schema, ranges, missing values, duplicates, and target leakage.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# Expected schema
REQUIRED_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "season",
    "cooperative_support",
    "yield_tons",
    "high_risk",
]

NUMERIC_COLUMNS = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "yield_tons",
]

CATEGORICAL_COLUMNS = ["season"]
BINARY_COLUMNS = ["cooperative_support", "high_risk"]

# Expected ranges based on generation rules
EXPECTED_RANGES = {
    "rainfall_mm": (600, 1400),
    "avg_temp_c": (12, 26),
    "soil_moisture_pct": (20, 80),
    "fertilizer_kg": (0, 250),
    "field_size_ha": (0.2, 4.0),
    "yield_tons": (0.5, None),  # minimum 0.5 after clipping
    "cooperative_support": (0, 1),
    "high_risk": (0, 1),
}

EXPECTED_SEASONS = {"A", "B"}

# Student-specific
LLLL = 7419
VERIFICATION_CODE = "AI-A2-7419"


def validate_seed(seed: int) -> bool:
    """Validate that the provided seed matches the student's LLLL."""
    return seed == LLLL


def check_required_columns(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check that all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    issues = []
    if missing:
        issues.append(f"Missing required columns: {missing}")
    if extra:
        issues.append(f"Unexpected extra columns: {extra}")
    return len(issues) == 0, issues


def check_ranges(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check that numeric values are within expected ranges."""
    issues = []
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        # Skip NaN values for range check
        valid_vals = df[col].dropna()
        if len(valid_vals) == 0:
            continue
        if min_val is not None and (valid_vals < min_val).any():
            issues.append(f"{col}: values below minimum {min_val}")
        if max_val is not None and (valid_vals > max_val).any():
            issues.append(f"{col}: values above maximum {max_val}")
    return len(issues) == 0, issues


def check_missing_values(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check missing value patterns - should only be in soil_moisture_pct and fertilizer_kg."""
    issues = []
    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            if col not in ["soil_moisture_pct", "fertilizer_kg"]:
                issues.append(f"Unexpected missing values in {col}: {count}")
            else:
                # Expected missing values
                pass
    return len(issues) == 0, issues


def check_duplicates(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check for duplicate rows."""
    issues = []
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        # Note: 3 duplicates are expected by design - this is a warning, not an error
        issues.append(f"Found {dup_count} duplicate rows (expected by design)")
    # Return True since expected duplicates are OK
    return True, issues


def check_target_leakage(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check for potential target leakage in features."""
    issues = []
    # high_risk is derived from yield_tons / field_size_ha
    # Neither yield_tons nor yield_per_ha should be used as features
    # This is a design-time check - we just verify the columns exist
    if "yield_tons" not in df.columns:
        issues.append("yield_tons column missing (regression target)")
    if "high_risk" not in df.columns:
        issues.append("high_risk column missing (classification target)")
    # Check that yield_per_ha is not a column (would be leakage)
    if "yield_per_ha" in df.columns:
        issues.append("yield_per_ha column present - potential target leakage")
    return len(issues) == 0, issues


def check_season_values(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check that season column only contains A or B."""
    issues = []
    if "season" in df.columns:
        invalid = set(df["season"].dropna().unique()) - EXPECTED_SEASONS
        if invalid:
            issues.append(f"Invalid season values: {invalid}")
    return len(issues) == 0, issues


def check_binary_columns(df: pd.DataFrame) -> Tuple[bool, list]:
    """Check that binary columns only contain 0 or 1."""
    issues = []
    for col in BINARY_COLUMNS:
        if col in df.columns:
            invalid = set(df[col].dropna().unique()) - {0, 1}
            if invalid:
                issues.append(f"Invalid values in {col}: {invalid}")
    return len(issues) == 0, issues


def validate_dataset(df: pd.DataFrame, seed: int) -> Tuple[bool, list]:
    """
    Run all validation checks on the dataset.
    Returns (is_valid, list_of_issues).
    """
    all_issues = []
    
    # Seed validation
    if not validate_seed(seed):
        all_issues.append(f"Invalid seed: expected {LLLL}, got {seed}")
    
    # Run all checks
    checks = [
        check_required_columns,
        check_ranges,
        check_missing_values,
        check_duplicates,
        check_target_leakage,
        check_season_values,
        check_binary_columns,
    ]
    
    for check_func in checks:
        valid, issues = check_func(df)
        all_issues.extend(issues)
    
    # Filter out expected issues (warnings)
    expected_issues = [i for i in all_issues if "expected by design" in i]
    actual_issues = [i for i in all_issues if "expected by design" not in i]
    
    if expected_issues:
        for issue in expected_issues:
            print(f"  [WARNING] {issue}")
    
    is_valid = len(actual_issues) == 0
    return is_valid, actual_issues


def load_and_validate(csv_path: Path, seed: int) -> pd.DataFrame:
    """Load CSV and validate it. Raises ValueError if validation fails."""
    df = pd.read_csv(csv_path)
    is_valid, issues = validate_dataset(df, seed)
    if not is_valid:
        raise ValueError(f"Dataset validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
    print(f"[OK] Dataset validation passed ({len(df)} rows)")
    return df


if __name__ == "__main__":
    # Test validation on generated data
    csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        valid, issues = validate_dataset(df, LLLL)
        if valid:
            print("Validation PASSED")
        else:
            print("Validation FAILED:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print(f"CSV not found at {csv_path}")