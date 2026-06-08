"""
Objective 3: Feedback Loop Update Function

Purpose
-------
This script demonstrates the feedback loop architecture for VTC's matching system.

It does NOT duplicate the Objective 2 optimiser. Instead, it:
1. Consumes recent match_events data.
2. Aggregates behaviour by client-trainer pair.
3. Converts behavioural events into match_outcome labels.
4. Builds a temporary optimiser-ready dataset.
5. Calls the existing Objective 2 weight_optimisation.py script.
6. Leaves the generated weight profile ready to be written into weight_profiles.

Expected flow:
    match_events
    -> feedback aggregation
    -> Client-Trainer_Match_Result.csv
    -> weight_optimisation.py
    -> outputs/optimised_weight_profile.json
    -> weight_profiles table

This is designed as an architecture prototype. In production, the input would come
from Supabase match_events, and the final JSON would be written back to the
weight_profiles table by the VTC backend/dev team.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Simulated or exported match_events data.
MATCH_EVENTS_PATH = Path("Feedback_Architecture/match_events_simulation.csv")

# Existing Objective 2 optimiser.
# Update this path if your optimiser is inside another folder.
OPTIMISER_SCRIPT = Path("Weight_optimization/weight_optimisation.py")

# The current optimiser expects this filename as input.
OPTIMISER_INPUT_PATH = Path("Weight_optimization/Client-Trainer_Match_Result.csv")

# Audit output folder for Objective 3.
OUTPUT_DIR = Path("outputs_objective3")

# Minimum activation thresholds.
MIN_EVENT_ROWS = 500
MIN_SUCCESSFUL_OUTCOMES = 30


# ============================================================
# EVENT AND FEATURE MAPPING
# ============================================================

EVENT_PRIORITY: Dict[str, int] = {
    "impression": 0,
    "profile_open": 1,
    "chat_tap": 2,
    "booking_start": 3,
    "booking_confirm": 4,
}

# For the architecture design, booking_start and booking_confirm are treated
# as successful outcomes. chat_tap remains a strong engagement signal but not
# a final success label.
SUCCESS_EVENTS = {"booking_start", "booking_confirm"}

# match_events stores subscores using production factor names.
# weight_optimisation.py expects *_score column names.
SUBSCORE_TO_OPTIMISER_COLUMNS: Dict[str, str] = {
    "gender": "gender_score",
    "style": "style_score",
    "avail": "availability_score",
    "age": "age_score",
    "qual": "qualification_score",
    "edu": "education_score",
    "loc": "location_score",
    "recency": "recency_score",
    "exp": "experience_score",
    "persona": "persona_score",
    "goal": "goal_score",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_subscores(value: str | dict) -> Dict[str, float]:
    """
    Parse the subscores field from match_events.

    In production this may already be JSON. In local simulations it may appear
    as a string representation of a dictionary.
    """
    if isinstance(value, dict):
        return value

    try:
        return json.loads(value)
    except Exception:
        return ast.literal_eval(value)


def validate_match_events(events: pd.DataFrame) -> None:
    """Validate that the required match_events columns exist."""
    required_columns = ["client_id", "trainer_id", "event", "score", "subscores"]
    missing = [col for col in required_columns if col not in events.columns]

    if missing:
        raise ValueError(f"Missing required match_events columns: {missing}")

    unknown_events = set(events["event"].dropna().unique()) - set(EVENT_PRIORITY.keys())
    if unknown_events:
        raise ValueError(f"Unknown event types found: {unknown_events}")


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw match_events into one row per client-trainer pair.

    The strongest event reached is retained. For example:
        impression -> profile_open -> chat_tap -> booking_confirm
    becomes:
        booking_confirm
    """
    events = events.copy()
    events["event_rank"] = events["event"].map(EVENT_PRIORITY)

    aggregated = (
        events.sort_values(["client_id", "trainer_id", "event_rank"])
        .groupby(["client_id", "trainer_id"], as_index=False)
        .tail(1)
        .copy()
    )

    aggregated = aggregated.rename(columns={"event": "highest_event"})
    aggregated["match_outcome"] = aggregated["highest_event"].isin(SUCCESS_EVENTS).astype(int)

    return aggregated


def build_optimizer_dataset(aggregated: pd.DataFrame) -> pd.DataFrame:
    """
    Convert aggregated feedback rows into the same dataset format expected by
    weight_optimisation.py.
    """
    rows = []

    for _, row in aggregated.iterrows():
        subscores = parse_subscores(row["subscores"])

        output = {
            "client_id": row["client_id"],
            "trainer_id": row["trainer_id"],
            "match_outcome": int(row["match_outcome"]),
            "baseline_score_default_weights": float(row["score"]),
        }

        for subscore_key, output_column in SUBSCORE_TO_OPTIMISER_COLUMNS.items():
            output[output_column] = float(subscores.get(subscore_key, 0.0))

        rows.append(output)

    return pd.DataFrame(rows)


def check_activation_thresholds(events: pd.DataFrame, training_data: pd.DataFrame) -> None:
    """
    Prevent the update loop from running on too little behavioural data.
    """
    if len(events) < MIN_EVENT_ROWS:
        raise ValueError(
            f"Only {len(events)} match_events rows found. "
            f"Minimum required: {MIN_EVENT_ROWS}."
        )

    successful_outcomes = int(training_data["match_outcome"].sum())
    if successful_outcomes < MIN_SUCCESSFUL_OUTCOMES:
        raise ValueError(
            f"Only {successful_outcomes} successful outcomes found. "
            f"Minimum required: {MIN_SUCCESSFUL_OUTCOMES}."
        )


def trigger_weight_optimiser() -> None:
    """
    Run the existing Objective 2 optimiser.

    This keeps Objective 3 lightweight and avoids duplicated optimisation logic.
    """
    if not OPTIMISER_SCRIPT.exists():
        raise FileNotFoundError(
            f"{OPTIMISER_SCRIPT} was not found. "
            "Update OPTIMISER_SCRIPT to point to your weight_optimisation.py file."
        )

    subprocess.run([sys.executable, str(OPTIMISER_SCRIPT)], check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    events = pd.read_csv(MATCH_EVENTS_PATH)
    validate_match_events(events)

    aggregated = aggregate_events(events)
    training_data = build_optimizer_dataset(aggregated)

    # Save audit copy before triggering optimiser.
    feedback_dataset_path = OUTPUT_DIR / "feedback_training_dataset_for_optimizer.csv"
    training_data.to_csv(feedback_dataset_path, index=False)

    check_activation_thresholds(events, training_data)

    # Back up existing optimiser input so the update is reversible.
    if OPTIMISER_INPUT_PATH.exists():
        backup_path = OUTPUT_DIR / "previous_Client-Trainer_Match_Result_backup.csv"
        shutil.copy2(OPTIMISER_INPUT_PATH, backup_path)

    # Feed behavioural dataset into existing Objective 2 optimiser.
    training_data.to_csv(OPTIMISER_INPUT_PATH, index=False)

    print("Feedback training dataset prepared successfully.")
    print(f"Raw match_events rows: {len(events)}")
    print(f"Aggregated client-trainer pairs: {len(training_data)}")
    print(f"Successful behavioural outcomes: {int(training_data['match_outcome'].sum())}")
    print(f"Saved feedback dataset to: {feedback_dataset_path}")
    print("Triggering existing weight_optimisation.py...")

    trigger_weight_optimiser()

    print("Feedback loop update completed.")
    print("The updated weight profile should now be available in the optimiser output folder.")


if __name__ == "__main__":
    main()
