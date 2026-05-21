from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("Client-Trainer_Match_Result.csv")
OUTPUT_DIR = Path("outputs")

TARGET_COLUMN = "match_outcome"
THRESHOLD = 0.55
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
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1 values.")


def calculate_weighted_scores(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Calculate weighted average matching scores.

    The existing VTC matching algorithm uses:
        final_score = sum(weight_i * score_i) / sum(weight_i)
    """
    weight_sum = float(np.sum(weights))

    if weight_sum <= 0:
        return np.zeros(features.shape[0])

    return (features @ weights) / weight_sum


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Convert scores to predictions and return evaluation metrics."""
    predictions = (scores >= THRESHOLD).astype(int)

    return {
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),
    }


def optimisation_objective(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    """
    Objective function for optimisation.

    scipy minimises the returned value, so negative F1 is used to maximise F1.
    """
    scores = calculate_weighted_scores(features, weights)
    metrics = evaluate_predictions(labels, scores)
    return -metrics["f1"]


def export_weight_profile(weights: np.ndarray, output_path: Path) -> Dict[str, float]:
    """Export optimised weights using VTC weight profile key names."""
    profile = {
        WEIGHT_PROFILE_KEYS[feature]: round(float(weight), 4)
        for feature, weight in zip(FEATURE_COLUMNS, weights)
    }

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

    default_scores = calculate_weighted_scores(x_val, default_weights)
    default_metrics = evaluate_predictions(y_val, default_scores)

    # Weights are constrained to the same 0-5 range used by the admin UI sliders.
    bounds = [(0.0, 5.0)] * len(FEATURE_COLUMNS)

    optimisation_result = differential_evolution(
        optimisation_objective,
        bounds=bounds,
        args=(x_train, y_train),
        seed=RANDOM_STATE,
        maxiter=100,
        polish=True,
    )

    optimised_weights = optimisation_result.x
    optimised_scores = calculate_weighted_scores(x_val, optimised_weights)
    optimised_metrics = evaluate_predictions(y_val, optimised_scores)

    metrics_df = pd.DataFrame(
        [
            {"model": "Default weights", **default_metrics},
            {"model": "Optimised weights", **optimised_metrics},
        ]
    )

    metrics_path = OUTPUT_DIR / "weight_optimisation_metrics.csv"
    profile_path = OUTPUT_DIR / "optimised_weight_profile.json"

    metrics_df.to_csv(metrics_path, index=False)
    profile = export_weight_profile(optimised_weights, profile_path)

    print("Weight optimisation completed successfully.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Optimised weight profile saved to: {profile_path}")
    print("\nOptimised Weight Profile:")
    print(json.dumps(profile, indent=2))
    print("\nValidation Metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()