"""
Individual data generator for Musanze Harvest Intelligence Service.
Generates reproducible dataset using student-specific seed (LLLL = 7419).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Student-specific constants
LLLL = 7419
VERIFICATION_CODE = "AI-A2-7419"
BASE_ROWS = 240 + (LLLL % 41)  # 276 rows
MISSING_COUNT = round(0.05 * BASE_ROWS)  # 14 missing per column
DUPLICATE_ROWS = 3


def generate_base_data(rng: np.random.Generator) -> pd.DataFrame:
    """Generate base dataset with clean values."""
    n = BASE_ROWS
    
    # Generate features
    rainfall_mm = rng.uniform(600, 1400, n)
    avg_temp_c = rng.uniform(12, 26, n)
    soil_moisture_pct = rng.uniform(20, 80, n)
    fertilizer_kg = rng.uniform(0, 250, n)
    field_size_ha = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    cooperative_support = rng.choice([0, 1], n)
    
    # Season effect
    season_effect = np.where(season == "A", 0.50, -0.20)
    
    # Yield formula
    yield_tons = (
        0.004 * rainfall_mm
        + 0.06 * soil_moisture_pct
        + 0.02 * fertilizer_kg
        + 1.30 * field_size_ha
        - 0.25 * np.abs(avg_temp_c - 18)
        + 0.70 * cooperative_support
        + season_effect
        + rng.normal(0, 0.80, n)
    )
    
    # Clip at minimum 0.50 tons
    yield_tons = np.maximum(yield_tons, 0.50)
    
    # High risk: yield per hectare below median
    yield_per_ha = yield_tons / field_size_ha
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)
    
    df = pd.DataFrame({
        "rainfall_mm": rainfall_mm,
        "avg_temp_c": avg_temp_c,
        "soil_moisture_pct": soil_moisture_pct,
        "fertilizer_kg": fertilizer_kg,
        "field_size_ha": field_size_ha,
        "season": season,
        "cooperative_support": cooperative_support,
        "yield_tons": yield_tons,
        "high_risk": high_risk,
    })
    
    return df


def introduce_data_problems(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce deliberate missing values and duplicates."""
    df = df.copy()
    n = len(df)
    
    # Choose unique positions for soil_moisture_pct missing values
    soil_missing_idx = rng.choice(n, size=MISSING_COUNT, replace=False)
    df.loc[soil_missing_idx, "soil_moisture_pct"] = np.nan
    
    # Choose unique positions for fertilizer_kg missing values
    fert_missing_idx = rng.choice(n, size=MISSING_COUNT, replace=False)
    df.loc[fert_missing_idx, "fertilizer_kg"] = np.nan
    
    # Append 3 exact duplicate rows (duplicate the first 3 rows)
    duplicates = df.iloc[:DUPLICATE_ROWS].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df


def compute_fingerprint(csv_path: Path) -> str:
    """Compute SHA-256 fingerprint of the final CSV file."""
    with open(csv_path, "rb") as f:
        content = f.read()
    return hashlib.sha256(content).hexdigest()


def main():
    """Generate dataset and save artifacts."""
    print(f"Verification code: {VERIFICATION_CODE}")
    print(f"Seed (LLLL): {LLLL}")
    print(f"Base rows: {BASE_ROWS}")
    print(f"Missing values per column: {MISSING_COUNT}")
    print(f"Duplicate rows: {DUPLICATE_ROWS}")
    
    # Create reproducible RNG
    rng = np.random.default_rng(LLLL)
    
    # Generate data
    df_clean = generate_base_data(rng)
    df_final = introduce_data_problems(df_clean, rng)
    
    # Save CSV with 4 decimal places
    output_dir = Path(__file__).parent
    csv_path = output_dir / "AI_A2_2527419.csv"
    df_final.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved dataset to {csv_path} with {len(df_final)} rows")
    
    # Compute and save fingerprint
    fingerprint = compute_fingerprint(csv_path)
    fingerprint_path = output_dir.parent / "artifacts" / "data_fingerprint.txt"
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fingerprint_path, "w") as f:
        f.write(fingerprint)
    print(f"Data fingerprint (SHA-256): {fingerprint}")
    print(f"Saved fingerprint to {fingerprint_path}")
    
    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"Total rows: {len(df_final)}")
    print(f"Missing soil_moisture_pct: {df_final['soil_moisture_pct'].isna().sum()}")
    print(f"Missing fertilizer_kg: {df_final['fertilizer_kg'].isna().sum()}")
    print(f"Duplicate rows: {df_final.duplicated().sum()}")
    print(f"Yield range: {df_final['yield_tons'].min():.4f} - {df_final['yield_tons'].max():.4f}")
    print(f"High risk count: {df_final['high_risk'].sum()} ({df_final['high_risk'].mean()*100:.1f}%)")
    
    return VERIFICATION_CODE, fingerprint


if __name__ == "__main__":
    main()