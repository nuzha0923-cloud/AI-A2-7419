# Complete Run Instructions for AI_A2_2527419

## Project Location
```
C:\Users\Ali\Desktop\submit\AI_A2_2527419\
```

## Prerequisites
- Python 3.12 installed (tested version)
- Git (optional, for commit hash)

## Step 1: Create and Activate Virtual Environment
If the terminal prompt currently shows `(.venv)`, leave that environment first:
```powershell
deactivate
```
From the project folder, run this creation command only if `.venv312` does not
already exist:
```powershell
py -3.12 -m venv .venv312
```
Activate `.venv312`:
```powershell
.\.venv312\Scripts\Activate.ps1
python --version
```
Confirm the output is Python 3.12. Then install the project dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Step 2: Run Full Pipeline
```powershell
python run_all.py --student-id YOUR_EMAIL@ines.ac.rw --seed 7419
```
Replace `YOUR_EMAIL@ines.ac.rw` with your actual INES email.

**Expected output:**
- Verification code: AI-A2-7419
- Dataset generated with 279 rows
- All models trained and saved
- Artifacts created in `artifacts/`
- Models saved in `models/`
- Plots saved in `artifacts/plots/`

## Step 3: Test CLI Prediction
```powershell
python predict.py --json samples\valid.json
```
**Expected output:** JSON with yield_estimate_tons, high_risk_prediction, high_risk_probability, cluster, verification_code

```powershell
python predict.py --json samples\invalid.json
```
**Expected output:** Error message (validation fails)

## Step 4: Run FastAPI Server
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```
Keep this running. In another terminal:
- Open http://127.0.0.1:8000/docs
- Test `/health` endpoint
- Test `/model-info` endpoint
- Test `/predict` with valid and invalid inputs

## Step 5: Run Tests
```powershell
python -m pytest -q
```
**Expected:** 22 tests passing

## Step 6: Create Submission Package
```powershell
# From parent directory
cd C:\Users\Ali\Desktop
Compress-Archive -Path AI_A2_2527419 -DestinationPath AI_A2_2527419.zip
```
**Submit:** `AI_A2_2527419.zip` via INES e-learning platform

## Required Evidence Files (Create Before Submission)
1. **`evidence/AI_A2_2527419_DEMO.mp4`** - 2 min demo video (see VIDEO_GUIDE.md)
2. **`evidence/AI_A2_2527419_TEST_LOG.pdf`** - Export pytest output to PDF
3. **`evidence/AI_A2_2527419_DECLARATION.pdf`** - Fill and sign declaration template
4. **`evidence/AI_USE.md`** - Already created
5. **`docs/AI_A2_2527419_REPORT.pdf`** - 5-7 page technical report (see REPORT_TEMPLATE.md)

## Key Files to Understand for Defense
| File | Purpose |
|------|---------|
| `data/generate_data.py` | Dataset generation with seed 7419 |
| `src/validation.py` | Data validation & leakage checks |
| `src/preprocessing.py` | Train-only fitting pipeline |
| `src/regression.py` | Regression models + selection |
| `src/classification.py` | Classification models + selection |
| `src/clustering.py` | KMeans clustering + profiles |
| `src/api.py` | FastAPI endpoints |
| `run_all.py` | Main entry point |
| `predict.py` | CLI predictor |
| `tests/test_system.py` | All tests |

## Assessor Verification Commands
The assessor may run:
```powershell
# 1. Verify identity and fingerprint
python run_all.py --student-id YOUR_EMAIL@ines.ac.rw --seed 7419

# 2. Check fingerprint matches
type artifacts\data_fingerprint.txt

# 3. Check metrics
type artifacts\regression_metrics.json
type artifacts\classification_metrics.json
type artifacts\clustering_report.json

# 4. Test CLI
python predict.py --json samples\valid.json
python predict.py --json samples\invalid.json

# 5. Test API
uvicorn app:app --host 127.0.0.1 --port 8000
# Then test /health, /model-info, /predict

# 6. Run tests
python -m pytest -q

# 7. Ask code questions and request modification
```

## Common Issues & Fixes

### "ModuleNotFoundError"
```powershell
python -m pip install -r requirements.txt
```
Make sure `.venv312` is activated and `python --version` reports Python 3.12.

### "Seed mismatch" error
Ensure you use `--seed 7419` (your LLLL)

### "Model not found" error
Run `run_all.py` first to generate models

### Port 8000 in use
```powershell
uvicorn app:app --host 127.0.0.1 --port 8001
```

## Your Unique Values
- **LLLL (Seed):** 7419
- **Verification Code:** AI-A2-7419
- **Base Rows:** 276
- **Missing per column:** 14
- **Duplicates:** 3
- **Registration:** 25/27419
- **ZIP Name:** AI_A2_2527419.zip

## Deadline
**Monday, 19 October 2026 at 5:00 PM CAT (Africa/Kigali)**
Moodle timestamp is authoritative.

## Support
If you encounter issues:
1. Check the error message carefully
2. Ensure virtual environment is activated
3. Ensure all dependencies installed
4. Re-run `run_all.py` from clean state if needed