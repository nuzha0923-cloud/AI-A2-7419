"""
Clustering module for Musanze Harvest Intelligence Service.
Tests K from 2 to 6, uses silhouette score to justify K,
writes plain-language profile for each selected cluster.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS,
    create_preprocessing_pipeline,
    fit_preprocessor,
    transform_data,
)
from src.validation import VERIFICATION_CODE, LLLL


from src.preprocessing import (
    FEATURE_COLUMNS,
    create_preprocessing_pipeline,
    fit_preprocessor,
    transform_data,
    CATEGORICAL_COLUMNS,
)
from src.validation import VERIFICATION_CODE, LLLL


def scale_features_for_clustering(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Scale features for clustering using StandardScaler."""
    # One-hot encode categorical columns
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 7)) -> Dict[str, Any]:
    """
    Test K from 2 to 6, use silhouette score to justify K.
    Returns silhouette scores and optimal K.
    """
    scores = {}
    models = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=LLLL, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = float(score)
        models[k] = kmeans
        print(f"K={k}: Silhouette Score = {score:.4f}")
    
    optimal_k = max(scores, key=scores.get)
    print(f"\n[OK] Optimal K = {optimal_k} (Silhouette = {scores[optimal_k]:.4f})")
    
    return {
        "silhouette_scores": scores,
        "optimal_k": optimal_k,
        "models": models,
    }


def create_cluster_profiles(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Create plain-language profiles for each cluster.
    Uses original (unscaled) feature values for interpretability.
    """
    X_with_cluster = X.copy()
    X_with_cluster["cluster"] = labels
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_names if c not in numeric_cols]
    
    profiles = []
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        cluster_data = X_with_cluster[X_with_cluster["cluster"] == cluster_id]
        n_samples = len(cluster_data)
        
        # Compute statistics for each feature
        profile = {
            "cluster_id": int(cluster_id),
            "n_farms": int(n_samples),
            "percentage": float(n_samples / len(X) * 100),
            "feature_means": {},
            "feature_stds": {},
            "interpretation": "",
        }
        
        for feat in feature_names:
            if feat in cluster_data.columns:
                if feat in numeric_cols:
                    profile["feature_means"][feat] = float(cluster_data[feat].mean())
                    profile["feature_stds"][feat] = float(cluster_data[feat].std())
                else:
                    # For categorical, store mode
                    mode_val = cluster_data[feat].mode()
                    profile["feature_means"][feat] = mode_val.iloc[0] if len(mode_val) > 0 else None
                    profile["feature_stds"][feat] = None
        
        # Generate plain-language interpretation
        profile["interpretation"] = generate_cluster_interpretation(profile, cluster_id, n_clusters)
        profiles.append(profile)
    
    return profiles


def generate_cluster_interpretation(profile: Dict[str, Any], cluster_id: int, n_clusters: int) -> str:
    """Generate plain-language interpretation of a cluster."""
    means = profile["feature_means"]
    n = profile["n_farms"]
    
    # Compare to overall averages (will be filled in by caller)
    parts = [f"Cluster {cluster_id} contains {n} farms."]
    
    # Rainfall
    rain = means.get("rainfall_mm", 0)
    if rain > 1000:
        parts.append("These farms receive high rainfall (>1000 mm).")
    elif rain < 800:
        parts.append("These farms receive relatively low rainfall (<800 mm).")
    else:
        parts.append("These farms receive moderate rainfall (800-1000 mm).")
    
    # Temperature
    temp = means.get("avg_temp_c", 0)
    if temp > 22:
        parts.append("Average temperatures are warm (>22°C).")
    elif temp < 16:
        parts.append("Average temperatures are cool (<16°C).")
    else:
        parts.append("Average temperatures are moderate (16-22°C).")
    
    # Soil moisture
    moisture = means.get("soil_moisture_pct", 0)
    if moisture > 60:
        parts.append("Soil moisture is high (>60%).")
    elif moisture < 40:
        parts.append("Soil moisture is low (<40%).")
    else:
        parts.append("Soil moisture is moderate (40-60%).")
    
    # Fertilizer
    fert = means.get("fertilizer_kg", 0)
    if fert > 150:
        parts.append("Fertilizer use is high (>150 kg).")
    elif fert < 50:
        parts.append("Fertilizer use is low (<50 kg).")
    else:
        parts.append("Fertilizer use is moderate (50-150 kg).")
    
    # Field size
    size = means.get("field_size_ha", 0)
    if size > 2.5:
        parts.append("Farms are large (>2.5 ha).")
    elif size < 1.0:
        parts.append("Farms are small (<1 ha).")
    else:
        parts.append("Farms are medium-sized (1-2.5 ha).")
    
    # Season
    season_a = means.get("season_A", 0)  # This will be 0 or 1 after encoding
    if season_a > 0.7:
        parts.append("Predominantly Season A.")
    elif season_a < 0.3:
        parts.append("Predominantly Season B.")
    else:
        parts.append("Mixed seasons.")
    
    # Cooperative support
    coop = means.get("cooperative_support", 0)
    if coop > 0.7:
        parts.append("Most farms receive cooperative support.")
    elif coop < 0.3:
        parts.append("Few farms receive cooperative support.")
    else:
        parts.append("Some farms receive cooperative support.")
    
    return " ".join(parts)


def run_clustering_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete clustering experiment:
    1. Prepare features (exclude targets)
    2. Scale features
    3. Test K=2..6 with silhouette score
    4. Select optimal K
    5. Create cluster profiles
    6. Save results and model
    """
    # Prepare features (exclude targets)
    X = df[FEATURE_COLUMNS].copy()
    
    # Handle missing values for clustering (use median imputation on numeric columns only)
    from sklearn.impute import SimpleImputer
    numeric_cols = [c for c in FEATURE_COLUMNS if c != "season"]
    imputer = SimpleImputer(strategy="median")
    X_numeric = X[numeric_cols].copy()
    X_numeric_imputed = imputer.fit_transform(X_numeric)
    X_numeric_imputed_df = pd.DataFrame(X_numeric_imputed, columns=numeric_cols, index=X.index)
    
    # Combine with categorical column
    X_imputed_df = pd.concat([X_numeric_imputed_df, X[["season"]]], axis=1)
    
    # Scale features
    X_scaled, scaler = scale_features_for_clustering(X_imputed_df)
    
    # Find optimal K
    k_results = find_optimal_k(X_scaled)
    optimal_k = k_results["optimal_k"]
    best_kmeans = k_results["models"][optimal_k]
    
    # Get cluster labels
    labels = best_kmeans.labels_
    
    # Create profiles using original features
    profiles = create_cluster_profiles(X_imputed_df, labels, FEATURE_COLUMNS)
    
    # Save cluster assignments to dataframe for later use
    df_with_clusters = df.copy()
    df_with_clusters["cluster"] = labels
    
    # Save model with proper preprocessing pipeline
    # We need a ColumnTransformer that handles numeric imputation + scaling and categorical encoding
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    numeric_features = numeric_cols
    categorical_features = ["season"]
    
    # Create preprocessing for clustering
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])
    
    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])
    
    # Fit preprocessor on the data
    preprocessor.fit(X)
    
    # Save the full pipeline
    cluster_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("kmeans", best_kmeans),
    ])
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cluster_pipeline, models_dir / "clustering_pipeline.joblib")
    
    # Save cluster assignments
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster labels for test predictions
    np.save(artifacts_dir / "cluster_labels.npy", labels)
    
    clustering_results = {
        "verification_code": VERIFICATION_CODE,
        "seed": LLLL,
        "silhouette_scores": k_results["silhouette_scores"],
        "selected_k": optimal_k,
        "selected_silhouette": k_results["silhouette_scores"][optimal_k],
        "cluster_profiles": profiles,
        "n_clusters": optimal_k,
    }
    
    with open(artifacts_dir / "clustering_report.json", "w") as f:
        json.dump(clustering_results, f, indent=2)
    
    print(f"\n[OK] Clustering complete: K={optimal_k}")
    for p in profiles:
        print(f"  Cluster {p['cluster_id']}: {p['n_farms']} farms ({p['percentage']:.1f}%) - {p['interpretation']}")
    
    return {
        "pipeline": cluster_pipeline,
        "labels": labels,
        "profiles": profiles,
        "optimal_k": optimal_k,
        "X_scaled": X_scaled,
        "X_imputed_df": X_imputed_df,
    }


def plot_clusters(X_scaled: np.ndarray, labels: np.ndarray, output_path: Path):
    """Create 2D PCA visualization of clusters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=LLLL)
    X_2d = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7, s=30)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Farm Clusters (PCA Visualization)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Saved clusters plot to {output_path}")


if __name__ == "__main__":
    import pandas as pd
    csv_path = Path(__file__).parent.parent / "data" / "AI_A2_2527419.csv"
    df = pd.read_csv(csv_path)
    results = run_clustering_experiment(df)
    plot_clusters(results["X_scaled"], results["labels"], Path(__file__).parent.parent / "artifacts" / "plots" / "clusters.png")