# explainability.py
# Reads all 11 factor scores from the dataset and generates a plain English
# explanation for why a trainer was recommended to a client.
#
# The 11 matching factors (from Client-Trainer_Match_Result.csv):
#   1.  goal_score          : fitness goal alignment
#   2.  style_score         : training style compatibility
#   3.  persona_score       : coaching personality fit
#   4.  gender_score        : gender preference
#   5.  availability_score  : schedule overlap
#   6.  age_score           : client within trainer's age range
#   7.  qualification_score : trainer certifications
#   8.  education_score     : trainer education background
#   9.  location_score      : proximity for in-person sessions
#  10.  recency_score       : how recently the trainer's profile was updated
#  11.  experience_score    : trainer experience level

import json
from functools import lru_cache
import pandas as pd

CSV_PATH     = "Weight_optimization/Client-Trainer_Match_Result.csv"
EXCEL_PATH   = "Semantic_matching/semantic_matching__client_trainer_dataset.xlsx"
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

# Fallback priorities if the weights file is missing.
# Normally these are loaded directly from the optimised weights JSON.
FALLBACK_PRIORITY = {
    "goal_score"          : 2.5,
    "style_score"         : 3.0,
    "availability_score"  : 4.6,
    "persona_score"       : 3.8,
    "qualification_score" : 4.5,
    "experience_score"    : 3.2,
    "gender_score"        : 1.6,
    "age_score"           : 1.6,
    "location_score"      : 1.8,
    "recency_score"       : 2.6,
    "education_score"     : 0.1,
}

# For each factor: what to say when it's strong, okay, or weak.
# None means skip that tier (not worth mentioning).
FACTOR_PHRASES = {
    "goal_score": (
        lambda t: f"your fitness goals align with their focus on {t.get('goal_tags', 'their key areas')}",
        lambda t: "your fitness goals partially overlap with their focus areas",
        lambda t: "your fitness goals don't closely match their focus areas",
    ),
    "style_score": (
        lambda t: "your training style preferences are a strong match for what they offer",
        lambda t: "there is some overlap in preferred training styles",
        lambda t: "your preferred training style doesn't closely match what they offer",
    ),
    "persona_score": (
        lambda t: "their coaching personality is a good fit for your preferences",
        lambda t: "their coaching personality is a reasonable fit for what you're looking for",
        lambda t: "their coaching style may not fully match your personality preference",
    ),
    "qualification_score": (
        lambda t: "they hold the certifications you're looking for",
        lambda t: "their qualifications partially match what you're looking for",
        lambda t: "their certifications may not fully match your preference",
    ),
    "gender_score": (
        lambda t: "they match your gender preference",
        None,
        lambda t: "they don't match your stated gender preference",
    ),
    "availability_score": (
        lambda t: "your schedules overlap well",
        lambda t: "there is some schedule overlap",
        lambda t: "limited schedule overlap may make booking sessions difficult",
    ),
    "experience_score": (
        lambda t: "their experience level is exactly what you're looking for",
        lambda t: "their experience level is a reasonable fit",
        lambda t: "their experience level may not match your preference",
    ),
    "age_score": (
        lambda t: "you're within their typical client age range",
        None,
        lambda t: "you may be outside the age group they usually work with",
    ),
    "location_score": (
        lambda t: "they're conveniently located for in-person sessions",
        lambda t: "they're within a reasonable distance",
        lambda t: "distance may be a factor for in-person sessions",
    ),
    "recency_score": (
        lambda t: "their profile and availability are up to date",
        None,
        lambda t: "their profile hasn't been updated recently",
    ),
    "education_score": (
        lambda t: "their education background meets your preference",
        lambda t: "their education is a reasonable match",
        lambda t: "their education level may not align with your preference",
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


@lru_cache(maxsize=None)
def load_weight_profile():
    """
    Loads optimised weights and decision threshold from JSON.
    Returns (weights dict, threshold float, priority dict).
    Falls back to defaults if file is missing.
    """
    try:
        with open(WEIGHTS_PATH) as f:
            profile = json.load(f)

        threshold = float(profile.get("decision_threshold", 0.55))

        # reverse map: w_goals -> goal_score etc.
        key_to_col = {v: k for k, v in SCORE_TO_WEIGHT.items()}

        weights   = {}
        priority  = {}
        for key, val in profile.items():
            if key == "decision_threshold":
                continue
            col = key_to_col.get(key)
            if col:
                weights[col]  = float(val)
                priority[col] = float(val)  # use actual weight as priority

        return weights, threshold, priority

    except FileNotFoundError:
        return {}, 0.55, FALLBACK_PRIORITY


def compute_final_score(scores):
    # use optimised weights if available, otherwise fall back to simple average
    weights, _, _ = load_weight_profile()
    if weights:
        total, weighted = 0.0, 0.0
        for col in SCORE_TO_WEIGHT:
            w = weights.get(col, 1.0)
            weighted += w * scores.get(col, 0.0)
            total += w
        return round(weighted / total, 4) if total else 0.0
    vals = list(scores.values())
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def explain_match(trainer, scores):
    """
    Builds a plain English explanation using all 11 factors.
    Factor priorities and verdict threshold are loaded from the optimised weights JSON
    so the explanation stays in sync with the model automatically.
    """
    _, threshold, priority = load_weight_profile()

    positives    = []
    concerns     = []
    strong_count = 0

    for factor, (strong_fn, okay_fn, weak_fn) in FACTOR_PHRASES.items():
        val = scores.get(factor, 0)
        pri = priority.get(factor, 1.0)

        if val >= STRONG:
            strong_count += 1
            if strong_fn:
                positives.append((pri * val, strong_fn(trainer)))
        elif val >= OKAY and okay_fn:
            positives.append((pri * val * 0.5, okay_fn(trainer)))

        if val < WEAK and weak_fn:
            concerns.append((val, weak_fn(trainer)))

    positives.sort(key=lambda x: x[0], reverse=True)
    concerns.sort(key=lambda x: x[0])

    top_positives = [p for _, p in positives[:3]]
    top_concerns  = [c for _, c in concerns[:3]]

    # verdict uses the learned threshold from weight optimisation, not a hardcoded value
    final = scores.get("final_score", 0)
    if final >= threshold:
        verdict = "Strong match."
    elif final >= threshold * 0.75:
        verdict = "Decent match."
    else:
        verdict = "Closest available match."

    summary = f"{strong_count} of the 11 factors line up well."

    if top_positives:
        if len(top_positives) == 1:
            joined = top_positives[0]
        elif len(top_positives) == 2:
            joined = top_positives[0] + " and " + top_positives[1]
        else:
            joined = ", ".join(top_positives[:-1]) + ", and " + top_positives[-1]
        reason_text = "This trainer stands out because " + joined + "."
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
