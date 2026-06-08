from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("Weight_optimization/Client-Trainer_Match_Result.csv")
OUTPUT_DIR = Path("outputs")

TARGET_COLUMN = "match_outcome"
DEFAULT_THRESHOLD = 0.55
RANDOM_STATE = 42

FEATURE_COLUMNS: List[str] = [
    "gender_score",
    "style_score",
    "availability_score",
    "age_score",
    "qualification_score",
    "education_score",
    "location_score",
    "recency_score",
    "experience_score",
    "persona_score",
    "goal_score",
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "gender_score": 1.0,
    "style_score": 1.5,
    "availability_score": 1.0,
    "age_score": 0.8,
    "qualification_score": 1.2,
    "education_score": 0.5,
    "location_score": 1.0,
    "recency_score": 1.0,
    "experience_score": 1.0,
    "persona_score": 0.7,
    "goal_score": 1.5,
}

WEIGHT_PROFILE_KEYS: Dict[str, str] = {
    "gender_score": "w_gender",
    "style_score": "w_style",
    "availability_score": "w_avail",
    "age_score": "w_age",
    "qualification_score": "w_quals",
    "education_score": "w_edu",
    "location_score": "w_loc",
    "recency_score": "w_recency",
    "experience_score": "w_exp",
    "persona_score": "w_persona",
    "goal_score": "w_goals",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate required columns and basic label format."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    unique_labels = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1 values.")


def calculate_weighted_scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Calculate weighted average matching scores.

    This matches the VTC scoring structure:
        final_score = sum(weight_i * score_i) / sum(weight_i)
    """
    weight_sum = float(np.sum(weights))

    if weight_sum <= 0:
        return np.zeros(features.shape[0])

    return (features @ weights) / weight_sum


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """Convert continuous match scores to binary predictions and return metrics."""
    predictions = (scores >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),
    }


def optimisation_objective(
    params: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Optimise both factor weights and the decision threshold.

    The first values in params are the feature weights.
    The final value in params is the threshold used to classify a pair as successful.

    scipy minimises the returned value, so negative F1 is used to maximise F1.
    """
    weights = params[:-1]
    threshold = float(params[-1])

    scores = calculate_weighted_scores(features, weights)
    metrics = evaluate_predictions(labels, scores, threshold=threshold)

    combined_score = (
    0.40 * metrics["f1"] +
    0.25 * metrics["accuracy"] +
    0.20 * metrics["precision"] +
    0.15 * metrics["recall"]
)

    return -combined_score


def export_weight_profile(
    weights: np.ndarray,
    threshold: float,
    output_path: Path,
) -> Dict[str, float]:
    """
    Export optimised weights using VTC weight profile key names.

    The threshold is included for documentation because it is part of the trained
    decision rule used during validation. The current VTC weight_profiles table may
    only store the weight keys, so threshold deployment may need to be handled
    separately if used in production.
    """
    profile = {
        WEIGHT_PROFILE_KEYS[feature]: round(float(weight), 4)
        for feature, weight in zip(FEATURE_COLUMNS, weights)
    }
    profile["decision_threshold"] = round(float(threshold), 4)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return profile


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)

    features = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = df[TARGET_COLUMN].to_numpy(dtype=int)

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    default_weights = np.array(
        [DEFAULT_WEIGHTS[column] for column in FEATURE_COLUMNS],
        dtype=float,
    )

    # Baseline uses the current default weights and existing fixed threshold.
    default_scores = calculate_weighted_scores(x_val, default_weights)
    default_metrics = evaluate_predictions(
        y_val,
        default_scores,
        threshold=DEFAULT_THRESHOLD,
    )

    # Factor weights follow the 0-5 admin UI slider range.
    # The final parameter is the classification threshold.
    bounds = [(0.0, 5.0)] * len(FEATURE_COLUMNS)
    bounds.append((0.3, 0.8))

    optimisation_result = differential_evolution(
        optimisation_objective,
        bounds=bounds,
        args=(x_train, y_train),
        seed=RANDOM_STATE,
        maxiter=100,
        polish=True,
    )

    optimised_weights = optimisation_result.x[:-1]
    optimised_threshold = float(optimisation_result.x[-1])

    optimised_scores = calculate_weighted_scores(x_val, optimised_weights)
    optimised_metrics = evaluate_predictions(
        y_val,
        optimised_scores,
        threshold=optimised_threshold,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Default weights",
                "threshold": round(DEFAULT_THRESHOLD, 4),
                **default_metrics,
            },
            {
                "model": "Optimised weights + threshold",
                "threshold": round(optimised_threshold, 4),
                **optimised_metrics,
            },
        ]
    )

    metrics_path = OUTPUT_DIR / "weight_optimisation_metrics.csv"
    profile_path = OUTPUT_DIR / "optimised_weight_profile.json"

    metrics_df.to_csv(metrics_path, index=False)
    profile = export_weight_profile(optimised_weights, optimised_threshold, profile_path)

    print("Weight optimisation completed successfully.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Optimised weight profile saved to: {profile_path}")
    print(f"Optimised threshold: {optimised_threshold:.4f}")
    print("\nOptimised Weight Profile:")
    print(json.dumps(profile, indent=2))
    print("\nValidation Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
