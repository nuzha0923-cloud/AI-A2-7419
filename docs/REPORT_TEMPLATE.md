# Technical Report Template
# AI_A2_2527419_REPORT.pdf
# 5-7 pages, readable at 100% zoom

## Page 1: Identity and Decision Problem
- Full name : Nuzha Zainalabdeen Mohamed Ismail
- Registration number: 25/27419
- Verification code: AI-A2-7419
- Stakeholder: Musanze potato cooperative field officer
- Prediction questions:
  1. What is the expected yield (tons) for a given farm?
  2. Is this farm at high production risk?
  3. Which group of similar farms does this farm belong to?
- Ethical boundary statement

## Page 2: Data and Reproducibility
- Data dictionary (all 9 columns with types, ranges, meanings)
- Personal seed rule: LLLL=7419, numpy.random.default_rng(7419)
- Row count: 279 base + 3 duplicates = 282 total
- Missing values: 15 in soil_moisture_pct, 14 in fertilizer_kg
- Duplicates: 3 exact duplicate rows
- SHA-256 fingerprint: [from artifacts/data_fingerprint.txt]
- Dataset file: data/AI_A2_2527419.csv

## Page 3: Pipeline Design
- Train/test split: 80/20 stratified on high_risk, random_state=7419
- Preprocessing flow:
  1. ColumnTransformer with numeric (impute+scale), categorical (one-hot), binary (passthrough)
  2. Fit ONLY on training data
  3. Transform both train and test
- Leakage controls: No target columns in features; yield_per_ha not computed as feature
- Model candidates:
  - Regression: LinearRegression, Ridge, RandomForest, GradientBoosting
  - Classification: LogisticRegression, RandomForest, GradientBoosting
  - Clustering: KMeans K=2..6 with silhouette

## Page 4: Regression and Classification
- Regression metrics table (baseline + candidates + selected)
- Residual plot interpretation
- One important regression error analysis
- Classification metrics table (baseline + candidates + selected)
- Confusion matrix interpretation
- One important classification error analysis
- Model selection justifications

## Page 5: Clustering
- Silhouette scores for K=2..6
- Selected K with justification
- Cluster profiles (plain language)
- Operational meaning for field officer

## Page 6: API and Testing
- Endpoint contracts (GET /health, GET /model-info, POST /predict)
- Valid request/response example
- Invalid request examples (missing field, wrong type, range error, invalid season)
- Test summary (6+ tests passing)
- Failure behavior description

## Page 7: Model Card and Reflection
- Intended use
- Limitations (synthetic data, small sample, etc.)
- Risks
- Human oversight requirement
- What to improve with real data
