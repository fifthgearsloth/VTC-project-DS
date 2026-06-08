import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# ==============================
# CONFIG
# ==============================

DATA_PATH = "Semantic_matching/semantic_matching__client_trainer_dataset.xlsx"
MODEL_NAME = "all-MiniLM-L6-v2"

WEIGHTS = {
    "goal": 1.5,
    "style": 1.5,
    "persona": 0.7,
}

PERSONA_TRAITS_WEIGHT = 0.7
PERSONA_BIO_WEIGHT = 0.3


# ==============================
# HELPERS
# ==============================

def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def weighted_score(goal_score: float, style_score: float, persona_score: float) -> float:
    numerator = (
        WEIGHTS["goal"] * goal_score
        + WEIGHTS["style"] * style_score
        + WEIGHTS["persona"] * persona_score
    )
    denominator = WEIGHTS["goal"] + WEIGHTS["style"] + WEIGHTS["persona"]
    return numerator / denominator if denominator > 0 else 0.0


def clamp_similarity_matrix(sim_matrix):
    """
    Cosine similarity can sometimes produce tiny negative values.
    Clamp all values into [0, 1] for matching consistency.
    """
    sim_matrix[sim_matrix < 0] = 0.0
    sim_matrix[sim_matrix > 1] = 1.0
    return sim_matrix


# ==============================
# TEXT PREPARATION
# ==============================

def prepare_text_columns(clients: pd.DataFrame, trainers: pd.DataFrame):
    client_goals = clients["goal_tags"].fillna("").apply(clean_text).tolist()
    trainer_goals = trainers["goal_tags"].fillna("").apply(clean_text).tolist()

    client_styles = clients["styles_desired"].fillna("").apply(clean_text).tolist()
    trainer_styles = trainers["styles_offered"].fillna("").apply(clean_text).tolist()

    client_persona = clients["personality_traits"].fillna("").apply(clean_text).tolist()
    trainer_persona = trainers["personality_traits"].fillna("").apply(clean_text).tolist()

    trainer_bio = trainers["bio"].fillna("").apply(clean_text).tolist()

    return (
        client_goals,
        trainer_goals,
        client_styles,
        trainer_styles,
        client_persona,
        trainer_persona,
        trainer_bio,
    )


# ==============================
# SEMANTIC MATCHING
# ==============================

def run_semantic_matching() -> pd.DataFrame:
    print("Loading dataset...")
    clients = pd.read_excel(DATA_PATH, sheet_name="clients")
    trainers = pd.read_excel(DATA_PATH, sheet_name="trainers")

    print("Preparing text columns...")
    (
        client_goals,
        trainer_goals,
        client_styles,
        trainer_styles,
        client_persona,
        trainer_persona,
        trainer_bio,
    ) = prepare_text_columns(clients, trainers)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding goal texts...")
    client_goal_emb = model.encode(client_goals, convert_to_numpy=True, show_progress_bar=True)
    trainer_goal_emb = model.encode(trainer_goals, convert_to_numpy=True, show_progress_bar=True)

    print("Encoding style texts...")
    client_style_emb = model.encode(client_styles, convert_to_numpy=True, show_progress_bar=True)
    trainer_style_emb = model.encode(trainer_styles, convert_to_numpy=True, show_progress_bar=True)

    print("Encoding persona texts...")
    client_persona_emb = model.encode(client_persona, convert_to_numpy=True, show_progress_bar=True)
    trainer_persona_emb = model.encode(trainer_persona, convert_to_numpy=True, show_progress_bar=True)
    trainer_bio_emb = model.encode(trainer_bio, convert_to_numpy=True, show_progress_bar=True)

    print("Computing similarity matrices...")
    goal_sim = clamp_similarity_matrix(cosine_similarity(client_goal_emb, trainer_goal_emb))
    style_sim = clamp_similarity_matrix(cosine_similarity(client_style_emb, trainer_style_emb))
    persona_trait_sim = clamp_similarity_matrix(cosine_similarity(client_persona_emb, trainer_persona_emb))
    persona_bio_sim = clamp_similarity_matrix(cosine_similarity(client_persona_emb, trainer_bio_emb))

    print("Building rankings...")
    results = []

    for i, client in clients.iterrows():
        for j, trainer in trainers.iterrows():
            goal_score = float(goal_sim[i][j])
            style_score = float(style_sim[i][j])

            persona_traits_component = float(persona_trait_sim[i][j])
            persona_bio_component = float(persona_bio_sim[i][j])

            persona_score = (
                PERSONA_TRAITS_WEIGHT * persona_traits_component
                + PERSONA_BIO_WEIGHT * persona_bio_component
            )

            final_score = weighted_score(goal_score, style_score, persona_score)

            results.append({
                "client_id": client["client_id"],
                "trainer_id": trainer["trainer_id"],
                "goal_score": round(goal_score, 4),
                "style_score": round(style_score, 4),
                "persona_score": round(persona_score, 4),
                "persona_traits_component": round(persona_traits_component, 4),
                "persona_bio_component": round(persona_bio_component, 4),
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


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("Running semantic matching...")

    rankings = run_semantic_matching()

    print("Saving output files...")
    rankings.to_csv("semantic_full_results.csv", index=False)
    rankings[rankings["rank"] <= 3].to_csv("semantic_top3_results.csv", index=False)

    print("Saved:")
    print("- semantic_full_results.csv")
    print("- semantic_top3_results.csv")

    print("\nTop 10 preview:\n")
    print(rankings[rankings["rank"] <= 3].head(10))
