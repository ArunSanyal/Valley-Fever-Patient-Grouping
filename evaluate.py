
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)

from preprocess import preprocess
from cluster import fit_final_model

UNCERTAIN_THRESHOLD = 0.05


def build_mixed_distance_matrix(df, categorical_indices, gamma):

    matrix = np.asarray(df, dtype=object)
    n_cols = matrix.shape[1]
    numeric_idx = [i for i in range(n_cols) if i not in categorical_indices]

    numeric_matrix = matrix[:, numeric_idx].astype(float)
    numeric_dist = squareform(pdist(numeric_matrix, metric="sqeuclidean"))

    n = matrix.shape[0]
    categorical_dist = np.zeros((n, n))
    for i in categorical_indices:
        col = matrix[:, i].astype(str)
        categorical_dist += (col[:, None] != col[None, :]).astype(float)

    return numeric_dist + gamma * categorical_dist


def compute_silhouette_scores(matrix, categorical_indices, k_range=range(2, 8)):
   
    results = []
    for k in k_range:
        model, labels = fit_final_model(matrix, categorical_indices, k)
        dist_matrix = build_mixed_distance_matrix(matrix, categorical_indices, model.gamma)
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        results.append((k, score))
    return results


def cluster_profiles(processed_df, labels, original_df):
    
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in processed_df.columns if c not in numeric_cols]

    df = original_df.copy()
    df["cluster_label"] = labels

    binary_cols = [
        c for c in categorical_cols if df[c].dropna().isin([True, False]).all()
    ]
    mode_cols = [c for c in categorical_cols if c not in binary_cols]

    profiles = []
    for cluster_id, group in df.groupby("cluster_label"):
        profile = {"cluster_label": cluster_id, "n_patients": len(group)}
        for col in numeric_cols:
            profile[col] = group[col].mean()
        for col in binary_cols:
            profile[col] = group[col].astype(bool).mean()
        for col in mode_cols:
            profile[col] = group[col].mode().iloc[0]
        profiles.append(profile)

    return pd.DataFrame(profiles).set_index("cluster_label")


if __name__ == "__main__":
    processed_df, categorical_indices, scaler = preprocess()
    matrix = processed_df.to_numpy()

    # unscaled original values, needed for profiles, ARI/NMI, and the saved output
    full_df = pd.read_csv("synthetic_patients.csv")
    original_df = full_df.drop(columns=["patient_id"])

    print("Computing silhouette scores for k=2..7...")
    k_scores = compute_silhouette_scores(matrix, categorical_indices, k_range=range(2, 8))
    for k, score in k_scores:
        print(f"  k={k}: silhouette={score:.4f}")

    best_k, best_score = max(k_scores, key=lambda pair: pair[1])
    print(f"\nBest k by silhouette score: k={best_k} (score={best_score:.4f})")
    if best_k != 4:
        print(
            f"Note: this differs from the k=4 elbow-based guess in cluster.py -- "
            f"silhouette score is the more reliable signal, so we go with k={best_k}."
        )

    print(f"\nFitting final model with k={best_k}...")
    model, labels = fit_final_model(matrix, categorical_indices, best_k)
    dist_matrix = build_mixed_distance_matrix(matrix, categorical_indices, model.gamma)

    # measures recovery of the synthetic archetypes, not real-world validity (see README.md)
    ari = adjusted_rand_score(full_df["ground_truth_archetype"], labels)
    nmi = normalized_mutual_info_score(full_df["ground_truth_archetype"], labels)
    print(f"\nAdjusted Rand Index vs. ground_truth_archetype: {ari:.4f}")
    print(f"Normalized Mutual Info vs. ground_truth_archetype: {nmi:.4f}")

    # per-patient silhouette, so borderline patients can be flagged individually
    sample_silhouette = silhouette_samples(dist_matrix, labels, metric="precomputed")
    uncertain = sample_silhouette < UNCERTAIN_THRESHOLD
    print(
        f"\nPatients flagged uncertain/borderline (silhouette < {UNCERTAIN_THRESHOLD}): "
        f"{uncertain.sum()} / {len(uncertain)}"
    )

    print("\nCluster profiles (human-readable):")
    profile_df = cluster_profiles(processed_df, labels, original_df)
    print(profile_df.to_string())

    output_df = full_df.drop(columns=["ground_truth_archetype"]).copy()
    output_df["cluster_label"] = labels
    output_df["silhouette_value"] = sample_silhouette
    output_df["uncertain"] = uncertain
    output_df.to_csv("clustered_patients.csv", index=False)
    print("\nSaved clustered_patients.csv (with silhouette_value / uncertain columns)")
