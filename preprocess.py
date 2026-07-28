
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = "synthetic_patients.csv"

# fever_present / erythema_nodosum are booleans but treated as categories, not numbers
CATEGORICAL_COLUMNS = [
    "housing_stability",
    "employment_status",
    "education_level",
    "insurance_status",
    "fever_present",
    "erythema_nodosum",
]


def load_data(csv_path=CSV_PATH):
    """Load patients and drop the columns we never cluster on: patient_id
    (just a label) and ground_truth_archetype (the answer key)."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["patient_id", "ground_truth_archetype"])
    return df


def get_categorical_indices(df, categorical_columns=CATEGORICAL_COLUMNS):

    return [df.columns.get_loc(col) for col in categorical_columns]


def preprocess(csv_path=CSV_PATH):
    
    df = load_data(csv_path)

    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    numeric_cols = [c for c in df.columns if c not in categorical_cols]

    # scale so no single numeric column (e.g. distance in miles) dominates distance
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_cols])
    scaled_numeric_df = pd.DataFrame(
        scaled_numeric, columns=numeric_cols, index=df.index
    )

    # cast to string since kmodes' matching distance expects string labels
    categorical_df = df[categorical_cols].astype(str)

    processed_df = pd.concat([scaled_numeric_df, categorical_df], axis=1)
    processed_df = processed_df[df.columns]

    categorical_indices = get_categorical_indices(processed_df, categorical_cols)

    return processed_df, categorical_indices, scaler


if __name__ == "__main__":
    processed_df, categorical_indices, scaler = preprocess()

    categorical_names = [processed_df.columns[i] for i in categorical_indices]
    numeric_names = [c for c in processed_df.columns if c not in categorical_names]

    print(f"Processed data shape: {processed_df.shape}\n")

    print(f"Categorical columns ({len(categorical_names)}): {categorical_names}")
    print(f"Categorical column indices: {categorical_indices}\n")

    print(f"Numeric columns ({len(numeric_names)}): {numeric_names}\n")

    print("Preview (first 3 rows):")
    print(processed_df.head(3))
