# Musanze Harvest Intelligence Service
**AI-A2-7419** | Individual Assignment 2 | SWE 3513 Artificial Intelligence

## Student Information
- **Full Name:** Nuzha Zainalabdeen Mohamed Ismail
- **Registration Number:** 25/27419
- **Institutional Username:** ug2527419@ines.ac.rw
- **Verification Code:** AI-A2-7419
- **Python Version:** 3.12+
- **Tested OS:** Windows 11

## Problem Statement
A potato cooperative in Musanze receives farm records in inconsistent spreadsheets. Its field officer needs an evidence-based tool that can:
1. **Estimate harvest quantity** (regression: yield in tons)
2. **Identify farms at high production risk** (classification: high_risk flag)
3. **Group farms with similar conditions** (clustering)

Internet access may be unreliable, so model training and command-line prediction must work locally. A simple API is required for later integration with a mobile application.

**Ethical Boundary:** This service is a classroom decision-support prototype. It must not claim to replace an agronomist or determine access to finance, insurance, or public services.

## Dataset
- **Source:** Synthetic, generated from registration number seed (LLLL=7419)
- **Size:** 279 base rows + 3 duplicates = 282 total rows
- **Features:** 7 (rainfall_mm, avg_temp_c, soil_moisture_pct, fertilizer_kg, field_size_ha, season, cooperative_support)
- **Targets:** yield_tons (regression), high_risk (classification)
- **Deliberate Issues:** 15 missing values in soil_moisture_pct, 14 in fertilizer_kg, 3 duplicate rows
- **Fingerprint (SHA-256):** See `artifacts/data_fingerprint.txt`

## Quick Start

### 1. Create Virtual Environment
Use Python 3.12 (the tested version). On Windows, create and activate a fresh
environment from the project folder. If the terminal prompt shows another
active environment, leave it first with `deactivate`. Run the creation command
only if `.venv312` does not already exist:
```powershell
py -3.12 -m venv .venv312
```
Then activate the environment:
```powershell
.\.venv312\Scripts\Activate.ps1
python --version
```
Confirm the version is Python 3.12 before continuing. Do not recreate an
environment while it is active.

On Linux/macOS, create and activate `.venv` with Python 3.12:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
python run_all.py --student-id ug2527419@ines.ac.rw --seed 7419
```
This will:
- Generate/recreate the dataset
- Validate data integrity
- Train regression, classification, and clustering models
- Save all models and artifacts
- Generate plots (residuals.png, confusion_matrix.png, clusters.png)
- Create test_predictions.csv

### 4. Command-Line Prediction
```bash
python predict.py --json samples/valid.json
```

### 5. Run FastAPI Server
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 6. Run Tests
```bash
python -m pytest -q
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Predict yield, risk, cluster |

### Example Request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall_mm": 950.0,
    "avg_temp_c": 18.5,
    "soil_moisture_pct": 55.0,
    "fertilizer_kg": 120.0,
    "field_size_ha": 1.5,
    "season": "A",
    "cooperative_support": 1
  }'
```

### Example Response
```json
{
  "yield_estimate_tons": 3.4215,
  "high_risk_prediction": 0,
  "high_risk_probability": 0.2341,
  "cluster": 2,
  "verification_code": "AI-A2-7419"
}
```

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Model Info
```bash
curl http://127.0.0.1:8000/model-info
```

## Artifacts Generated

```
artifacts/
├── data_fingerprint.txt          # SHA-256 of dataset
├── regression_metrics.json       # MAE, RMSE, R² for all models
├── classification_metrics.json   # Precision, Recall, F1, ROC-AUC, CM
├── clustering_report.json        # Silhouette scores, profiles
├── test_predictions.csv          # Test set predictions
└── plots/
    ├── residuals.png
    ├── confusion_matrix.png
    └── clusters.png

models/
├── regression_pipeline.joblib
├── classification_pipeline.joblib
└── clustering_pipeline.joblib
```

## Model Selection Summary

### Regression
- **Baseline:** Mean predictor (DummyRegressor)
- **Candidates:** LinearRegression, Ridge, RandomForest, GradientBoosting
- **Selection:** Best RMSE on held-out test set

### Classification
- **Baseline:** Majority class (DummyClassifier)
- **Candidates:** LogisticRegression, RandomForest, GradientBoosting
- **Selection:** Best F1 score on held-out test set

### Clustering
- **Method:** KMeans with StandardScaler
- **K Range:** 2-6
- **Selection:** Highest silhouette score
- **Profiles:** Plain-language interpretation per cluster

## Testing

Run all tests:
```bash
python -m pytest -q
```

Test coverage includes:
1. Valid inference (happy path)
2. Missing field validation
3. Wrong type validation
4. Range error validation
5. Reproducibility check
6. API response validation

## Known Limitations

1. **Synthetic Data:** All data is generated from a formula + noise; not real farm data
2. **Small Dataset:** Only ~280 records; models may not generalize to real farms
3. **No Temporal Dynamics:** Season is categorical; no time-series modeling
4. **Feature Engineering:** Limited to provided features; no soil type, variety, etc.
5. **Local Only:** No cloud deployment; models saved as joblib files
6. **Binary Risk:** High-risk threshold is median yield/ha; not agronomically validated

## Responsible AI Statement

**Intended Use:** Classroom prototype for decision support to potato cooperative field officers in Musanze.

**Limitations:**
- Synthetic data only; not validated on real farms
- Small sample size (n≈280)
- No uncertainty quantification beyond point predictions
- Clustering is unsupervised; clusters may not align with actionable groups

**Risks:**
- Over-reliance on predictions without agronomist review
- Potential bias from synthetic generation formula
- Missing real-world factors (pests, diseases, market prices)

**Prohibited Uses:**
- Determining access to finance, insurance, or government subsidies
- Replacing professional agronomic advice
- Automated decision-making without human oversight

**Human Review Required:** All predictions must be reviewed by a qualified agronomist before any operational decisions.

## Repository
- **URL:** https://github.com/<your-github-username>/AI-A2-7419.git
- **Final Commit Hash:** [set on final commit]

## Evidence Files (for submission)
- `evidence/AI_A2_2527419_DEMO.mp4` - 2-minute demo video
- `evidence/AI_A2_2527419_TEST_LOG.pdf` - pytest output
- `evidence/AI_A2_2527419_DECLARATION.pdf` - Academic integrity declaration
- `evidence/AI_USE.md` - AI assistance disclosure
- `docs/AI_A2_2527419_REPORT.md` - technical report

## Submission Package
```bash
# Create submission zip (from parent directory)
zip -r AI_A2_2527419.zip AI_A2_2526883/
# (project folder keeps its original name AI_A2_2526883; the zip uses the new ID)
```
Submit `AI_A2_2527419.zip` through INES e-learning platform.

---

**Deadline:** Monday, 19 October 2026 at 5:00 PM CAT (Africa/Kigali)
