import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# CONFIG
# ==============================

DATA_PATH = "semantic_matching__client_trainer_dataset.xlsx"

SEMANTIC_WEIGHT = 0.7
JACCARD_WEIGHT = 0.3


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def split_tags(text):
    if pd.isna(text):
        return []
    return [t.strip().lower() for t in str(text).split(",") if t.strip()]


# ==============================
# PROFILE BUILDERS 
# ==============================

def build_client_text(row):
    goals = clean_text(row.get("goal_tags"))
    styles = clean_text(row.get("styles_desired"))
    persona = clean_text(row.get("personality_traits"))

    return " ".join([
        f"client goals {goals}",
        f"wants coaching style {styles}",
        f"prefers trainer personality {persona}",
        f"needs {goals} training program",
    ])


def build_trainer_text(row):
    goals = clean_text(row.get("goal_tags"))
    styles = clean_text(row.get("styles_offered"))
    persona = clean_text(row.get("personality_traits"))
    bio = clean_text(row.get("bio"))

    return " ".join([
        f"trainer specialises in {goals}",
        f"coaching style includes {styles}",
        f"trainer personality is {persona}",
        f"{bio}",
    ])


# ==============================
# JACCARD BASELINE
# ==============================

def jaccard(a, b):
    set_a = set(split_tags(a))
    set_b = set(split_tags(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def jaccard_score(client, trainer):
    goal = jaccard(client["goal_tags"], trainer["goal_tags"])
    style = jaccard(client["styles_desired"], trainer["styles_offered"])
    persona = jaccard(client["personality_traits"], trainer["personality_traits"])

    return (0.5 * goal) + (0.3 * style) + (0.2 * persona)


# ==============================
# SEMANTIC MATCHER 
# ==============================

def semantic_match(clients, trainers):
    client_texts = clients.apply(build_client_text, axis=1).tolist()
    trainer_texts = trainers.apply(build_trainer_text, axis=1).tolist()

    corpus = client_texts + trainer_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),   
        max_features=5000
    )

    matrix = vectorizer.fit_transform(corpus)

    client_matrix = matrix[:len(client_texts)]
    trainer_matrix = matrix[len(client_texts):]

    similarity = cosine_similarity(client_matrix, trainer_matrix)

    return similarity


# ==============================
# HYBRID MATCHER
# ==============================

def run_matching():
    clients = pd.read_excel(DATA_PATH, sheet_name="clients")
    trainers = pd.read_excel(DATA_PATH, sheet_name="trainers")

    semantic_sim = semantic_match(clients, trainers)

    results = []

    for i, client in clients.iterrows():
        for j, trainer in trainers.iterrows():

            sem_score = float(semantic_sim[i][j])
            jac_score = jaccard_score(client, trainer)

            final_score = (SEMANTIC_WEIGHT * sem_score) + (JACCARD_WEIGHT * jac_score)

            results.append({
                "client_id": client["client_id"],
                "trainer_id": trainer["trainer_id"],
                "semantic_score": round(sem_score, 4),
                "jaccard_score": round(jac_score, 4),
                "final_score": round(final_score, 4),
            })

    df = pd.DataFrame(results)

    df = df.sort_values(["client_id", "final_score"], ascending=[True, False])

    df["rank"] = df.groupby("client_id")["final_score"]\
                   .rank(method="first", ascending=False)

    return df


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("Starting matcher...")

    rankings = run_matching()

    print("Finished computing rankings")

    print("Saving files...")

    rankings.to_csv("advanced_matcher_full.csv", index=False)
    rankings[rankings["rank"] <= 3].to_csv("advanced_matcher_top3.csv", index=False)

    print("Files saved successfully!")

    print("\nTop matches preview:\n")
    print(rankings[rankings["rank"] <= 3].head(10))