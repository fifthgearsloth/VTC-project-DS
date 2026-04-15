import pandas as pd  # type: ignore

# ==============================
# CONFIG
# ==============================

DATA_PATH = "semantic_matching__client_trainer_dataset.xlsx"

WEIGHTS = {
    "goal": 1.5,
    "style": 1.5,
    "persona": 0.7,
}


# ==============================
# HELPERS
# ==============================

def split_tags(text) -> list[str]:
    """Convert comma-separated text into cleaned tag list."""
    if pd.isna(text):
        return []
    return [t.strip().lower() for t in str(text).split(",") if t.strip()]


def jaccard(a, b) -> float:
    """Jaccard similarity = intersection / union."""
    set_a = set(split_tags(a))
    set_b = set(split_tags(b))

    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def weighted_score(scores: dict[str, float]) -> float:
    numerator = sum(WEIGHTS[k] * scores[k] for k in scores)
    denominator = sum(WEIGHTS[k] for k in scores)
    return numerator / denominator if denominator > 0 else 0.0


# ==============================
# FACTOR SCORING
# ==============================

def jaccard_factor_scores(client: pd.Series, trainer: pd.Series) -> dict[str, float]:
    goal_score = jaccard(client["goal_tags"], trainer["goal_tags"])
    style_score = jaccard(client["styles_desired"], trainer["styles_offered"])
    persona_score = jaccard(client["personality_traits"], trainer["personality_traits"])

    return {
        "goal": goal_score,
        "style": style_score,
        "persona": persona_score,
    }


# ==============================
# MAIN MATCHING
# ==============================

def run_jaccard_matching() -> pd.DataFrame:
    clients = pd.read_excel(DATA_PATH, sheet_name="clients")
    trainers = pd.read_excel(DATA_PATH, sheet_name="trainers")

    results = []

    for _, client in clients.iterrows():
        for _, trainer in trainers.iterrows():
            scores = jaccard_factor_scores(client, trainer)
            final_score = weighted_score(scores)

            results.append({
                "client_id": client["client_id"],
                "trainer_id": trainer["trainer_id"],
                "goal_score": round(scores["goal"], 4),
                "style_score": round(scores["style"], 4),
                "persona_score": round(scores["persona"], 4),
                "final_score": round(final_score, 4),
            })

    df = pd.DataFrame(results)
    df = df.sort_values(["client_id", "final_score"], ascending=[True, False]).copy()

    df["rank"] = (
        df.groupby("client_id")["final_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return df


if __name__ == "__main__":
    print("Running Jaccard baseline matching...")

    rankings = run_jaccard_matching()

    rankings.to_csv("jaccard_full_results.csv", index=False)
    rankings[rankings["rank"] <= 3].to_csv("jaccard_top3_results.csv", index=False)

    print("Saved:")
    print("- jaccard_full_results.csv")
    print("- jaccard_top3_results.csv")
    print("\nTop 10 preview:")
    print(rankings[rankings["rank"] <= 3].head(10))