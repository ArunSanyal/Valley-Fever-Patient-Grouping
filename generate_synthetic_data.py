"""
Synthetic Valley Fever (coccidioidomycosis) patient data generator.

WHY SYNTHETIC DATA
-------------------
Real Valley Fever patient records are protected health information and any
project touching them requires IRB approval and data-use agreements. This
mirrors the constraint Prof. Leroy's own lab operates under, which is why
their symptom-extraction work trains on privacy-preserving LLM-generated
synthetic clinical text rather than real notes. Here we apply the same
justification to structured (tabular) data instead of free text: we generate
statistically plausible patients so the clustering pipeline can be built,
tested, and demonstrated with zero privacy exposure and no IRB dependency.

WHY LATENT ARCHETYPES + NOISE (RATHER THAN FULLY RANDOM FEATURES)
-------------------------------------------------------------------
If every feature were drawn independently at random, "clusters" found later
would be statistical artifacts with no ground truth to check against -- we
could not tell a good clustering solution from a lucky one. Instead, this
generator samples most patients from one of a small number of hidden
"archetypes": realistic joint profiles of symptom severity + social
determinants of health (SDOH) that plausibly co-occur in practice (e.g.
severe symptoms tend to co-occur with poor care access in real populations,
not because one causes the other, but because both reflect the same
underlying deprivation). A `true_archetype` label is retained so that
cluster.py / evaluate.py can later check whether K-Prototypes actually
recovers this structure -- a genuine test of the method, not a foregone
conclusion.

A configurable fraction of patients (`noise_fraction`) is instead drawn with
EACH feature sampled independently from the overall population's marginal
distribution (a uniform mixture across archetypes), decoupling the joint
structure that defines an archetype. These patients don't cleanly belong to
any archetype -- exactly like real populations always contain atypical
individuals -- which keeps the clustering problem from being trivially easy
and gives the elbow-method / silhouette analysis something genuine to argue
about.

WHY `occupational_exposure_risk` (BEYOND THE BASE ASSIGNMENT PROMPT)
-----------------------------------------------------------------------
Coccidioides spores live in Southwestern soil and are inhaled when soil is
disturbed, so occupational exposure (construction, agriculture, military
ground work, landscaping) is a well-documented Valley Fever risk factor
specific to Arizona's epidemiology -- distinct from the generic SDOH domains
in the base assignment (housing, employment, education, social support,
access to care). Including it lets a cluster emerge that is symptomatic and
high-exposure but otherwise well-resourced (e.g. insured construction
workers) -- a group whose actionable intervention is proactive worksite
screening, not the social-services outreach appropriate for a low-resource
cluster. That distinction is the kind of precision-medicine, subgroup-specific
strategy Prof. Leroy's research aims to surface.

IMPORTANT: `true_archetype` is a hidden ground-truth label for OUR OWN
validation only (did clustering recover real structure?). It must NEVER be
passed as a feature into the clustering algorithm itself -- doing so would be
data leakage, since it is not something a clinician would know in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

SYMPTOM_COLUMNS = [
    "fatigue",
    "cough",
    "fever",
    "night_sweats",
    "chest_pain",
    "arthralgia",
    "headache",
    "rash",
    "weight_loss",
    "dyspnea",
]

CATEGORICAL_COLUMNS = [
    "housing_stability",
    "employment_status",
    "education_level",
    "social_support",
    "access_to_care",
    "occupational_exposure_risk",
]

CATEGORICAL_LEVELS = {
    "housing_stability": ["stable", "unstable", "homeless"],
    "employment_status": ["employed", "unemployed", "unable_to_work"],
    "education_level": ["less_than_hs", "hs_diploma", "some_college", "bachelor_plus"],
    "social_support": ["strong", "moderate", "weak"],
    "access_to_care": ["insured_regular_care", "insured_no_regular_care", "uninsured"],
    "occupational_exposure_risk": ["high", "low"],
}

# Human-readable name for each archetype index, used as the `true_archetype`
# ground-truth label. Order matters: it must match the keys used in the two
# parameter dicts below.
ARCHETYPE_NAMES = {
    0: "mild_well_supported",
    1: "severe_underserved",
    2: "high_exposure_resourced",
    3: "moderate_chronic_under_resourced_access",
}

# Per-archetype (mean, std) for each numeric symptom severity, on a 0-4 scale
# (none / mild / moderate / severe / very severe). Values are clipped to
# [0, 4] at draw time since a Gaussian can otherwise stray outside the scale.
ARCHETYPE_SYMPTOM_PARAMS = {
    0: {  # mild_well_supported: low severity across the board
        "fatigue": (0.8, 0.6), "cough": (0.6, 0.5), "fever": (0.4, 0.4),
        "night_sweats": (0.5, 0.5), "chest_pain": (0.4, 0.4), "arthralgia": (0.6, 0.5),
        "headache": (0.7, 0.5), "rash": (0.3, 0.4), "weight_loss": (0.4, 0.4),
        "dyspnea": (0.5, 0.5),
    },
    1: {  # severe_underserved: high severity across the board
        "fatigue": (3.4, 0.6), "cough": (3.2, 0.6), "fever": (3.0, 0.7),
        "night_sweats": (3.1, 0.6), "chest_pain": (2.8, 0.7), "arthralgia": (3.0, 0.6),
        "headache": (2.7, 0.7), "rash": (2.0, 0.8), "weight_loss": (2.9, 0.7),
        "dyspnea": (3.1, 0.6),
    },
    2: {  # high_exposure_resourced: respiratory/systemic symptoms from
          # occupational dust exposure dominate; skin/wasting symptoms don't
        "fatigue": (2.2, 0.6), "cough": (2.8, 0.5), "fever": (1.8, 0.6),
        "night_sweats": (1.6, 0.6), "chest_pain": (2.3, 0.6), "arthralgia": (1.4, 0.5),
        "headache": (1.5, 0.5), "rash": (0.8, 0.5), "weight_loss": (1.0, 0.5),
        "dyspnea": (2.5, 0.6),
    },
    3: {  # moderate_chronic_under_resourced_access: lingering fatigue/joint
          # pain/weight loss consistent with delayed or inconsistent care
        "fatigue": (2.6, 0.6), "cough": (1.8, 0.6), "fever": (1.5, 0.6),
        "night_sweats": (1.7, 0.6), "chest_pain": (1.6, 0.6), "arthralgia": (2.4, 0.6),
        "headache": (1.9, 0.6), "rash": (1.0, 0.5), "weight_loss": (2.2, 0.6),
        "dyspnea": (1.7, 0.6),
    },
}

# Per-archetype category probabilities for each SDOH / categorical feature.
# Each inner dict's values must sum to 1.0.
ARCHETYPE_CATEGORICAL_PARAMS = {
    0: {
        "housing_stability": {"stable": 0.85, "unstable": 0.12, "homeless": 0.03},
        "employment_status": {"employed": 0.80, "unemployed": 0.15, "unable_to_work": 0.05},
        "education_level": {"less_than_hs": 0.05, "hs_diploma": 0.25, "some_college": 0.35, "bachelor_plus": 0.35},
        "social_support": {"strong": 0.70, "moderate": 0.25, "weak": 0.05},
        "access_to_care": {"insured_regular_care": 0.80, "insured_no_regular_care": 0.15, "uninsured": 0.05},
        "occupational_exposure_risk": {"high": 0.15, "low": 0.85},
    },
    1: {
        "housing_stability": {"stable": 0.10, "unstable": 0.45, "homeless": 0.45},
        "employment_status": {"employed": 0.10, "unemployed": 0.55, "unable_to_work": 0.35},
        "education_level": {"less_than_hs": 0.45, "hs_diploma": 0.35, "some_college": 0.15, "bachelor_plus": 0.05},
        "social_support": {"strong": 0.05, "moderate": 0.25, "weak": 0.70},
        "access_to_care": {"insured_regular_care": 0.05, "insured_no_regular_care": 0.25, "uninsured": 0.70},
        "occupational_exposure_risk": {"high": 0.55, "low": 0.45},
    },
    2: {
        "housing_stability": {"stable": 0.70, "unstable": 0.25, "homeless": 0.05},
        "employment_status": {"employed": 0.75, "unemployed": 0.20, "unable_to_work": 0.05},
        "education_level": {"less_than_hs": 0.30, "hs_diploma": 0.45, "some_college": 0.20, "bachelor_plus": 0.05},
        "social_support": {"strong": 0.45, "moderate": 0.40, "weak": 0.15},
        "access_to_care": {"insured_regular_care": 0.55, "insured_no_regular_care": 0.35, "uninsured": 0.10},
        "occupational_exposure_risk": {"high": 0.90, "low": 0.10},
    },
    3: {
        "housing_stability": {"stable": 0.45, "unstable": 0.40, "homeless": 0.15},
        "employment_status": {"employed": 0.35, "unemployed": 0.35, "unable_to_work": 0.30},
        "education_level": {"less_than_hs": 0.20, "hs_diploma": 0.40, "some_college": 0.30, "bachelor_plus": 0.10},
        "social_support": {"strong": 0.25, "moderate": 0.50, "weak": 0.25},
        "access_to_care": {"insured_regular_care": 0.15, "insured_no_regular_care": 0.60, "uninsured": 0.25},
        "occupational_exposure_risk": {"high": 0.35, "low": 0.65},
    },
}

NOISE_LABEL = "noise"


# ---------------------------------------------------------------------------
# Per-patient feature draws
# ---------------------------------------------------------------------------

def _draw_numeric_from_archetype(archetype_idx: int, rng: np.random.Generator) -> dict:
    return {
        feat: float(np.clip(rng.normal(mean, std), 0, 4))
        for feat, (mean, std) in ARCHETYPE_SYMPTOM_PARAMS[archetype_idx].items()
    }


def _draw_categorical_from_archetype(archetype_idx: int, rng: np.random.Generator) -> dict:
    result = {}
    for feat, probs in ARCHETYPE_CATEGORICAL_PARAMS[archetype_idx].items():
        levels = list(probs.keys())
        weights = list(probs.values())
        result[feat] = rng.choice(levels, p=weights)
    return result


def _draw_noise_numeric(n_archetypes: int, rng: np.random.Generator) -> dict:
    # Each feature independently mixture-sampled across archetypes: no single
    # archetype governs the whole patient, so joint structure is destroyed
    # while each feature's own marginal distribution is preserved.
    result = {}
    for feat in SYMPTOM_COLUMNS:
        a = rng.integers(0, n_archetypes)
        mean, std = ARCHETYPE_SYMPTOM_PARAMS[a][feat]
        result[feat] = float(np.clip(rng.normal(mean, std), 0, 4))
    return result


def _draw_noise_categorical(n_archetypes: int, rng: np.random.Generator) -> dict:
    result = {}
    for feat in CATEGORICAL_COLUMNS:
        a = rng.integers(0, n_archetypes)
        probs = ARCHETYPE_CATEGORICAL_PARAMS[a][feat]
        levels = list(probs.keys())
        weights = list(probs.values())
        result[feat] = rng.choice(levels, p=weights)
    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_patients: int = 700,
    n_archetypes: int = 4,
    noise_fraction: float = 0.10,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Valley Fever patient dataset with latent archetype
    structure plus a fraction of unstructured "noise" patients.

    Parameters
    ----------
    n_patients : total number of synthetic patients to generate.
    n_archetypes : number of latent archetypes to draw structured patients
        from (must be <= the number of archetypes defined in
        ARCHETYPE_NAMES / ARCHETYPE_SYMPTOM_PARAMS / ARCHETYPE_CATEGORICAL_PARAMS).
    noise_fraction : fraction of patients (0-1) drawn with each feature
        independently sampled from the overall population marginal, rather
        than jointly from one archetype. See module docstring for rationale.
    random_seed : seed for the random generator, for reproducibility.

    Returns
    -------
    DataFrame with columns: patient_id, the 10 symptom severity columns,
    the 6 SDOH/categorical columns, and `true_archetype` (ground truth for
    validation only -- see module docstring; do not feed into clustering).
    """
    n_defined = len(ARCHETYPE_NAMES)
    if not (1 <= n_archetypes <= n_defined):
        raise ValueError(
            f"n_archetypes must be between 1 and {n_defined} "
            f"(the number of archetypes defined in this module), got {n_archetypes}."
        )
    if not (0.0 <= noise_fraction <= 1.0):
        raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}.")

    rng = np.random.default_rng(random_seed)

    n_noise = int(round(n_patients * noise_fraction))
    n_structured = n_patients - n_noise

    # Balanced (as evenly as possible) assignment of structured patients
    # across archetypes, so downstream value_counts / cluster-size sanity
    # checks aren't confounded by an accidental class imbalance.
    archetype_ids = list(range(n_archetypes))
    base_count = n_structured // n_archetypes
    remainder = n_structured % n_archetypes
    structured_assignment = np.repeat(archetype_ids, base_count)
    if remainder:
        extra = rng.choice(archetype_ids, size=remainder, replace=False)
        structured_assignment = np.concatenate([structured_assignment, extra])
    rng.shuffle(structured_assignment)

    rows = []

    for archetype_idx in structured_assignment:
        row = {}
        row.update(_draw_numeric_from_archetype(archetype_idx, rng))
        row.update(_draw_categorical_from_archetype(archetype_idx, rng))
        row["true_archetype"] = ARCHETYPE_NAMES[archetype_idx]
        rows.append(row)

    for _ in range(n_noise):
        row = {}
        row.update(_draw_noise_numeric(n_archetypes, rng))
        row.update(_draw_noise_categorical(n_archetypes, rng))
        row["true_archetype"] = NOISE_LABEL
        rows.append(row)

    df = pd.DataFrame(rows)

    # Shuffle so structured/noise patients aren't in contiguous blocks, then
    # assign patient IDs after shuffling so ID order carries no information.
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df.insert(0, "patient_id", [f"P{i:04d}" for i in range(1, len(df) + 1)])

    # Keep column order stable and readable: id, symptoms, SDOH, ground truth.
    df = df[["patient_id"] + SYMPTOM_COLUMNS + CATEGORICAL_COLUMNS + ["true_archetype"]]

    return df


if __name__ == "__main__":
    OUTPUT_PATH = "synthetic_patients.csv"

    data = generate_synthetic_data(
        n_patients=700,
        n_archetypes=4,
        noise_fraction=0.10,
        random_seed=42,
    )

    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(data)} synthetic patients to {OUTPUT_PATH}\n")

    print("Patient counts by true_archetype (ground truth, for validation only):")
    print(data["true_archetype"].value_counts())

    print("\nSummary statistics for numeric symptom severity columns:")
    print(data[SYMPTOM_COLUMNS].describe())
