# explainability.py
# Generates a plain English explanation for why a trainer was recommended.
# Based on the semantic match scores for goals, style, and personality.
# TODO: extend explanations below once D1 (full 11-factor baseline) is complete.
# Need to add explanation blocks for these 8 remaining factors,
# following the same if/elif/else pattern used above.
#
#   - qualifications  (how well trainer's certs match client preference)
#   - gender          (gender preference match)
#   - availability    (schedule overlap)
#   - experience      (years of experience vs client preference)
#   - age             (is client in trainer's served age range)
#   - location        (distance for in-person sessions)
#   - recency         (how recently trainer updated their profile/location)
#   - education       (trainer's education level vs client preference)

# Score thresholds to decide how good each factor match is.
# Feel free to adjust these if the explanations feel off.
STRONG = 0.65   # clearly a good match
OKAY   = 0.45   # somewhat matches
WEAK   = 0.30   # not really matching


def explain_match(trainer: dict, scores: dict) -> str:
    """
    Pass in a trainer profile and their scores, get back a readable explanation string.

    trainer should have keys: goal_tags, styles_offered, personality_traits
    scores should have keys: goal_score, style_score, persona_score, final_score
    """

    reasons  = []   # strong points worth highlighting
    warnings = []   # weak points worth flagging


    # How well do the client's goals match this trainer's specialisation
    goal = scores.get("goal_score", 0)

    if goal >= STRONG:
        reasons.append(
            f"your fitness goals align closely with this trainer's specialisation "
            f"in {trainer.get('goal_tags', 'their focus areas')}"
        )
    elif goal >= OKAY:
        reasons.append("your fitness goals partially overlap with what this trainer focuses on")
    else:
        warnings.append("your fitness goals don't closely match this trainer's specialisation")


    # How well does the preferred training style match what the trainer offers
    style = scores.get("style_score", 0)

    if style >= STRONG:
        reasons.append(
            f"your preferred training style matches well with what they offer "
            f"({trainer.get('styles_offered', 'their sessions')})"
        )
    elif style >= OKAY:
        reasons.append("there is some overlap in training style preferences")
    else:
        warnings.append("your preferred training style does not closely match what this trainer offers")


    # How well does the trainer's personality match what the client is looking for
    persona = scores.get("persona_score", 0)

    if persona >= STRONG:
        reasons.append("the trainer's coaching personality suits your stated preferences")
    elif persona >= OKAY:
        reasons.append("the trainer's personality is a reasonable fit for what you are looking for")
    else:
        warnings.append("the trainer's coaching style may not fully match your personality preference")


    # Pick an opening line based on the overall score
    final = scores.get("final_score", 0)

    if final >= STRONG:
        opening = "Strong match,"
    elif final >= OKAY:
        opening = "Decent match,"
    else:
        opening = "Closest available match,"


    # Combine everything into one readable sentence
    if reasons:
        explanation = opening + " " + ", and ".join(reasons) + "."
    else:
        explanation = opening + " this trainer is the closest available based on your profile."

    # Tack on any warnings at the end so the user knows where the gaps are
    if warnings:
        explanation += " Note: " + "; ".join(warnings) + "."

    return explanation


# TODO: extend explanations below once D1 (full 11-factor baseline) is complete.
# Need to add explanation blocks for these 8 remaining factors,
# following the same if/elif/else pattern used above.
#
#   - qualifications  (how well trainer's certs match client preference)
#   - gender          (gender preference match)
#   - availability    (schedule overlap)
#   - experience      (years of experience vs client preference)
#   - age             (is client in trainer's served age range)
#   - location        (distance for in-person sessions)
#   - recency         (how recently trainer updated their profile/location)
#   - education       (trainer's education level vs client preference)


# Run this file directly to quickly check the output looks right
if __name__ == "__main__":

    sample_trainer = {
        "trainer_id"        : "T001",
        "goal_tags"         : "weight loss, fat loss, body recomposition",
        "styles_offered"    : "HIIT, strength training, metabolic circuits",
        "personality_traits": "motivating, energetic, results-driven",
        "bio"               : "I help busy people lose weight through smart training."
    }

    sample_scores = {
        "goal_score"   : 0.82,
        "style_score"  : 0.74,
        "persona_score": 0.55,
        "final_score"  : 0.73
    }

    result = explain_match(sample_trainer, sample_scores)
    print(result)
