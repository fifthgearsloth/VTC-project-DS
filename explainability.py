# explainability.py
# Reads all 11 factor scores from the dataset and generates a plain English
# explanation for why a trainer was recommended to a client.

import json
import pandas as pd

CSV_PATH     = "Client-Trainer_Match_Result.csv"
EXCEL_PATH   = "semantic_matching__client_trainer_dataset.xlsx"
WEIGHTS_PATH = "outputs/optimised_weight_profile.json"

STRONG = 0.65
OKAY   = 0.45
WEAK   = 0.30

# Maps CSV score columns to their weight keys in the optimised profile JSON
SCORE_TO_WEIGHT = {
    "goal_score"          : "w_goals",
    "style_score"         : "w_style",
    "persona_score"       : "w_persona",
    "gender_score"        : "w_gender",
    "availability_score"  : "w_avail",
    "age_score"           : "w_age",
    "qualification_score" : "w_quals",
    "education_score"     : "w_edu",
    "location_score"      : "w_loc",
    "recency_score"       : "w_recency",
    "experience_score"    : "w_exp",
}

# Goals and style matter most to a real training match so they surface first.
# Higher number = shown first when multiple factors are strong.
FACTOR_PRIORITY = {
    "goal_score"          : 10,
    "style_score"         : 10,
    "availability_score"  : 8,
    "persona_score"       : 7,
    "qualification_score" : 6,
    "experience_score"    : 5,
    "gender_score"        : 4,
    "age_score"           : 4,
    "location_score"      : 3,
    "recency_score"       : 2,
    "education_score"     : 2,
}

# For each factor: what to say when it's strong, okay, or weak.
# None means skip that tier (not worth mentioning).
FACTOR_PHRASES = {
    "goal_score": (
        lambda t: f"your fitness goals align closely with their specialisation in {t.get('goal_tags', 'their focus areas')}",
        lambda t: "your fitness goals partially overlap with what this trainer focuses on",
        lambda t: "your fitness goals don't closely match this trainer's specialisation",
    ),
    "style_score": (
        lambda t: f"your preferred training style matches well with what they offer ({t.get('styles_offered', 'their sessions')})",
        lambda t: "there is some overlap in preferred training styles",
        lambda t: "your preferred training style doesn't closely match what this trainer offers",
    ),
    "persona_score": (
        lambda t: "the trainer's coaching personality suits your stated preferences",
        lambda t: "the trainer's personality is a reasonable fit for what you're looking for",
        lambda t: "the trainer's coaching style may not fully match your personality preference",
    ),
    "qualification_score": (
        lambda t: "the trainer holds certifications that match what you're looking for",
        lambda t: "the trainer's qualifications partially meet your preference",
        lambda t: "the trainer's certifications may not fully match your preference",
    ),
    "gender_score": (
        lambda t: "your gender preference is met",
        None,
        lambda t: "the trainer doesn't match your stated gender preference",
    ),
    "availability_score": (
        lambda t: "your schedules overlap well",
        lambda t: "there is some schedule overlap with this trainer",
        lambda t: "limited schedule overlap may make booking sessions difficult",
    ),
    "experience_score": (
        lambda t: "the trainer's experience level matches what you're looking for",
        lambda t: "the trainer's experience is a reasonable fit",
        lambda t: "the trainer's experience level may not match your preference",
    ),
    "age_score": (
        lambda t: "you fall within the age group this trainer typically works with",
        None,
        lambda t: "you may be outside the age group this trainer usually works with",
    ),
    "location_score": (
        lambda t: "the trainer is conveniently located for in-person sessions",
        lambda t: "the trainer is within a reasonable distance",
        lambda t: "distance may be a factor for in-person sessions",
    ),
    "recency_score": (
        lambda t: "the trainer's profile and availability are up to date",
        None,
        lambda t: "the trainer's profile hasn't been updated recently",
    ),
    "education_score": (
        lambda t: "the trainer's education background meets your preference",
        lambda t: "the trainer's education is a reasonable match",
        lambda t: "the trainer's education level may not align with your preference",
    ),
}


def load_scores(client_id, trainer_id):
    df = pd.read_csv(CSV_PATH)
    row = df[(df["client_id"] == client_id) & (df["trainer_id"] == trainer_id)]
    if row.empty:
        raise ValueError(f"No data found for {client_id} x {trainer_id}")
    row = row.iloc[0]
    scores = {col: round(float(row[col]), 4) for col in SCORE_TO_WEIGHT}
    scores["final_score"] = compute_final_score(scores)
    return scores


def load_trainer_profile(trainer_id):
    trainers = pd.read_excel(EXCEL_PATH, sheet_name="trainers")
    # CSV uses TRN-2001 format, Excel uses T001
    excel_id = "T" + trainer_id.split("-")[1][-3:] if trainer_id.startswith("TRN-") else trainer_id
    row = trainers[trainers["trainer_id"] == excel_id]
    if row.empty:
        raise ValueError(f"Trainer {trainer_id} not found in dataset")
    row = row.iloc[0]
    return {
        "trainer_id"        : trainer_id,
        "goal_tags"         : str(row.get("goal_tags", "")),
        "styles_offered"    : str(row.get("styles_offered", "")),
        "personality_traits": str(row.get("personality_traits", "")),
        "bio"               : str(row.get("bio", "")),
    }


def compute_final_score(scores):
    # use optimised weights if available, otherwise fall back to simple average
    try:
        with open(WEIGHTS_PATH) as f:
            weights = json.load(f)
        total, weighted = 0.0, 0.0
        for col, key in SCORE_TO_WEIGHT.items():
            w = weights.get(key, 1.0)
            weighted += w * scores.get(col, 0.0)
            total += w
        return round(weighted / total, 4) if total else 0.0
    except FileNotFoundError:
        vals = list(scores.values())
        return round(sum(vals) / len(vals), 4) if vals else 0.0


def explain_match(trainer, scores):
    """
    Builds a plain English explanation using all 11 factors.
    Prioritises goals and style when picking the top 3 highlights.
    Includes a summary count of how many factors are strong.
    """
    positives    = []
    concerns     = []
    strong_count = 0

    for factor, (strong_fn, okay_fn, weak_fn) in FACTOR_PHRASES.items():
        val      = scores.get(factor, 0)
        priority = FACTOR_PRIORITY.get(factor, 1)

        if val >= STRONG:
            strong_count += 1
            if strong_fn:
                positives.append((priority * val, strong_fn(trainer)))
        elif val >= OKAY and okay_fn:
            positives.append((priority * val * 0.5, okay_fn(trainer)))

        if val < WEAK and weak_fn:
            concerns.append((val, weak_fn(trainer)))

    # highest priority and score first, worst concerns first
    positives.sort(key=lambda x: x[0], reverse=True)
    concerns.sort(key=lambda x: x[0])

    top_positives = [p for _, p in positives[:3]]
    top_concerns  = [c for _, c in concerns[:3]]

    final = scores.get("final_score", 0)
    if final >= STRONG:
        verdict = "Strong match."
    elif final >= OKAY:
        verdict = "Decent match."
    else:
        verdict = "Closest available match."

    summary = f"{strong_count} out of 11 factors are a strong match."

    if top_positives:
        reason_text = "This trainer stands out because " + ", and ".join(top_positives) + "."
    else:
        reason_text = "This trainer is the closest available based on your overall profile."

    concern_text = ("Worth noting: " + "; ".join(top_concerns) + ".") if top_concerns else ""

    parts = [verdict, summary, reason_text]
    if concern_text:
        parts.append(concern_text)

    return " ".join(parts)


def explain_for_pair(client_id, trainer_id):
    """Entry point. Pass IDs, get back a full explanation."""
    scores  = load_scores(client_id, trainer_id)
    trainer = load_trainer_profile(trainer_id)
    return explain_match(trainer, scores)


def score_bar(val, width=20):
    filled = int(val * width)
    bar = "#" * filled + "-" * (width - filled)
    if val >= STRONG:
        tier = "STRONG"
    elif val >= OKAY:
        tier = "OK    "
    else:
        tier = "WEAK  "
    return f"[{bar}] {val:.2f}  {tier}"


if __name__ == "__main__":

    CLIENT_ID  = "CLT-1022"
    TRAINER_ID = "TRN-2014"

    print(f"\n{'='*55}")
    print(f"  VTC Explainability Report")
    print(f"  Client: {CLIENT_ID}   Trainer: {TRAINER_ID}")
    print(f"{'='*55}\n")

    scores  = load_scores(CLIENT_ID, TRAINER_ID)
    trainer = load_trainer_profile(TRAINER_ID)

    print("Factor Scores:")
    print(f"  {'Factor':<25} {'Score Bar'}")
    print(f"  {'-'*50}")
    for k, v in scores.items():
        if k == "final_score":
            continue
        print(f"  {k:<25} {score_bar(v)}")
    print(f"  {'-'*50}")
    print(f"  {'final_score':<25} {score_bar(scores['final_score'])}\n")

    print("Explanation:")
    print(f"  {explain_match(trainer, scores)}\n")
