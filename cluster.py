import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes

from preprocess import preprocess

RANDOM_STATE = 42


def find_best_k(matrix, categorical_indices, k_range=range(2, 8)):

    results = []
    for k in k_range:
        model = KPrototypes(n_clusters=k, init="Cao", random_state=RANDOM_STATE)
        model.fit_predict(matrix, categorical=categorical_indices)
        results.append((k, model.cost_))
    return results


def fit_final_model(matrix, categorical_indices, k):
    
    model = KPrototypes(n_clusters=k, init="Cao", random_state=RANDOM_STATE)
    labels = model.fit_predict(matrix, categorical=categorical_indices)
    return model, labels


if __name__ == "__main__":
    processed_df, categorical_indices, scaler = preprocess()
    matrix = processed_df.to_numpy()

    print("Finding best k (elbow method)...")
    k_costs = find_best_k(matrix, categorical_indices, k_range=range(2, 8))
    for k, cost in k_costs:
        print(f"  k={k}: cost={cost:.2f}")

    # just a starting guess, evaluate.py checks this against silhouette score
    chosen_k = 4
    print(f"\nFitting final model with k={chosen_k}...")
    model, labels = fit_final_model(matrix, categorical_indices, chosen_k)

    print("\nPatients per cluster:")
    print(pd.Series(labels).value_counts().sort_index())

    # save original-scale values, not the scaled matrix, so the CSV is readable
    original_df = pd.read_csv("synthetic_patients.csv").drop(
        columns=["ground_truth_archetype"]
    )
    output_df = original_df.copy()
    output_df["cluster_label"] = labels
    output_df.to_csv("clustered_patients.csv", index=False)
    print("\nSaved clustered_patients.csv")
