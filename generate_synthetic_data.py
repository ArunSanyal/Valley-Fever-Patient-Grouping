
import numpy as np
import pandas as pd

N = 200


rng = np.random.default_rng(seed=42)


ARCHETYPES = [
    {
        "name": "delayed_diagnosis_vulnerable",
        # uninsured + unstable housing -> hard to get care, longer to diagnose
        "n": 45,
        "housing_stability_weights": {"unstable": 0.50, "at_risk": 0.35, "stable": 0.15},
        "employment_status_weights": {
            "employed_outdoor": 0.40, "unemployed": 0.35,
            "employed_indoor": 0.15, "retired": 0.10,
        },
        "education_level_weights": {
            "less_than_hs": 0.35, "hs_grad": 0.35,
            "some_college": 0.20, "bachelors_plus": 0.10,
        },
        "insurance_status_weights": {"uninsured": 0.45, "underinsured": 0.40, "insured": 0.15},
        "social_support_mean": 4.5,
        "distance_mean": 22,
        "respiratory_severity_mean": 2.0,
        "fatigue_level_mean": 2.2,
        "joint_pain_mean": 1.4,
        "fever_prob": 0.60,
        "erythema_prob": 0.08,
        "symptom_duration_mean": 45,
        "prior_misdiagnosis_mean": 2.0,
    },
    {
        "name": "stable_well_managed",
        # insured + stable + indoor job -> caught early, mild, few misdiagnoses
        "n": 55,
        "housing_stability_weights": {"stable": 0.85, "at_risk": 0.12, "unstable": 0.03},
        "employment_status_weights": {
            "employed_indoor": 0.55, "employed_outdoor": 0.15,
            "retired": 0.20, "unemployed": 0.10,
        },
        "education_level_weights": {
            "bachelors_plus": 0.45, "some_college": 0.30,
            "hs_grad": 0.20, "less_than_hs": 0.05,
        },
        "insurance_status_weights": {"insured": 0.85, "underinsured": 0.13, "uninsured": 0.02},
        "social_support_mean": 7.5,
        "distance_mean": 8,
        "respiratory_severity_mean": 0.7,
        "fatigue_level_mean": 0.9,
        "joint_pain_mean": 0.6,
        "fever_prob": 0.35,
        "erythema_prob": 0.04,
        "symptom_duration_mean": 8,
        "prior_misdiagnosis_mean": 0.3,
    },
    {
        "name": "elderly_moderate_risk",
        # mostly retired, lower social support, joint pain can look like normal aging
        "n": 50,
        "housing_stability_weights": {"stable": 0.70, "at_risk": 0.25, "unstable": 0.05},
        "employment_status_weights": {
            "retired": 0.80, "employed_indoor": 0.10,
            "unemployed": 0.08, "employed_outdoor": 0.02,
        },
        "education_level_weights": {
            "hs_grad": 0.40, "some_college": 0.30,
            "less_than_hs": 0.20, "bachelors_plus": 0.10,
        },
        "insurance_status_weights": {"insured": 0.55, "underinsured": 0.40, "uninsured": 0.05},
        "social_support_mean": 3.5,
        "distance_mean": 15,
        "respiratory_severity_mean": 1.5,
        "fatigue_level_mean": 1.7,
        "joint_pain_mean": 1.8,
        "fever_prob": 0.45,
        "erythema_prob": 0.05,
        "symptom_duration_mean": 25,
        "prior_misdiagnosis_mean": 1.0,
    },
    {
        "name": "young_acute_high_exposure",
        # outdoor workers, heavy exposure -> hits hard and fast, more rash
        "n": 50,
        "housing_stability_weights": {"stable": 0.75, "at_risk": 0.20, "unstable": 0.05},
        "employment_status_weights": {
            "employed_outdoor": 0.70, "employed_indoor": 0.20,
            "unemployed": 0.08, "retired": 0.02,
        },
        "education_level_weights": {
            "hs_grad": 0.40, "some_college": 0.30,
            "less_than_hs": 0.20, "bachelors_plus": 0.10,
        },
        "insurance_status_weights": {"insured": 0.60, "underinsured": 0.30, "uninsured": 0.10},
        "social_support_mean": 6.5,
        "distance_mean": 14,
        "respiratory_severity_mean": 2.4,
        "fatigue_level_mean": 2.0,
        "joint_pain_mean": 1.2,
        "fever_prob": 0.75,
        "erythema_prob": 0.20,
        "symptom_duration_mean": 12,
        "prior_misdiagnosis_mean": 0.5,
    },
]


def sample_categorical(weights, n, rng):
    """Draw n values from a categorical distribution defined by weights."""
    categories = list(weights.keys())
    probs = list(weights.values())
    return rng.choice(categories, size=n, p=probs)


def sample_ordinal(mean, n, rng, std=0.7, low=0, high=3):
    """Draw n values for a 0-3 symptom score, centered on mean with some noise."""
    values = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(values.round(), low, high).astype(int)


def generate_cohort(archetypes=ARCHETYPES, rng=rng):
    blocks = []

    for arch in archetypes:
        n = arch["n"]

        housing_stability = sample_categorical(arch["housing_stability_weights"], n, rng)
        employment_status = sample_categorical(arch["employment_status_weights"], n, rng)
        education_level = sample_categorical(arch["education_level_weights"], n, rng)
        insurance_status = sample_categorical(arch["insurance_status_weights"], n, rng)

        # 0 = no support network, 10 = very supported
        social_support_score = np.clip(
            rng.normal(loc=arch["social_support_mean"], scale=2.0, size=n), 0, 10
        ).round().astype(int)

        # gamma distribution: right-skewed, most patients close, a few far out
        distance_to_clinic_miles = np.clip(
            rng.gamma(shape=2.0, scale=arch["distance_mean"] / 2.0, size=n), 1, 60
        ).round(1)

        respiratory_severity = sample_ordinal(arch["respiratory_severity_mean"], n, rng)
        fatigue_level = sample_ordinal(arch["fatigue_level_mean"], n, rng)
        joint_pain = sample_ordinal(arch["joint_pain_mean"], n, rng)

        fever_present = rng.random(n) < arch["fever_prob"]

        # classic Valley Fever rash, uncommon but distinctive (see README)
        erythema_nodosum = rng.random(n) < arch["erythema_prob"]

        # right-skewed again, most patients near the mean but with a long tail
        symptom_duration_days = np.clip(
            rng.gamma(shape=2.0, scale=arch["symptom_duration_mean"] / 2.0, size=n), 3, 120
        ).round().astype(int)

        # poisson fits "count of times misdiagnosed" well
        prior_misdiagnosis_count = np.clip(
            rng.poisson(lam=arch["prior_misdiagnosis_mean"], size=n), 0, 3
        )

        block = pd.DataFrame(
            {
                "respiratory_severity": respiratory_severity,
                "fatigue_level": fatigue_level,
                "fever_present": fever_present,
                "joint_pain": joint_pain,
                "erythema_nodosum": erythema_nodosum,
                "symptom_duration_days": symptom_duration_days,
                "prior_misdiagnosis_count": prior_misdiagnosis_count,
                "housing_stability": housing_stability,
                "employment_status": employment_status,
                "education_level": education_level,
                "social_support_score": social_support_score,
                "insurance_status": insurance_status,
                "distance_to_clinic_miles": distance_to_clinic_miles,
                # answer key for validation later, not a clustering input
                "ground_truth_archetype": arch["name"],
            }
        )
        blocks.append(block)

    cohort = pd.concat(blocks, ignore_index=True)

    # shuffle so archetype order doesn't leak through row order
    cohort = cohort.sample(frac=1, random_state=42).reset_index(drop=True)
    return cohort


def generate_dataset(n=N):
    """Glue everything together: archetype-based cohort plus patient IDs."""
    cohort = generate_cohort()
    assert len(cohort) == n, f"Archetype counts must sum to N={n}"

    patient_ids = [f"SYN-{i:04d}" for i in range(1, n + 1)]
    combined = pd.concat([pd.DataFrame({"patient_id": patient_ids}), cohort], axis=1)

    return combined


if __name__ == "__main__":
    df = generate_dataset(N)

    out_path = "synthetic_patients.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} synthetic patients -> {out_path}\n")

    print("Archetype balance (ground truth, not a clustering input):")
    print(df["ground_truth_archetype"].value_counts())

    print("\nPreview (first 5 rows):")
    print(df.head(5))

    print("\nNumeric column summary (mean / std):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].agg(["mean", "std"]).T)
