# generate_trainer_profiles.py
# Generates a rich trainer profile CSV with raw data for all 11 matching factors.
# Uses existing Excel text profiles + CSV score averages to create realistic values.
# Output: trainer_profiles_full.csv
#
# Run: python generate_trainer_profiles.py

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

EXCEL_PATH  = "semantic_matching__client_trainer_dataset.xlsx"
CSV_PATH    = "Client-Trainer_Match_Result.csv"
OUTPUT_PATH = "trainer_profiles_full.csv"

random.seed(42)
np.random.seed(42)

# raw value options
LOCATIONS = [
    "Melbourne CBD", "South Yarra", "Richmond", "Fitzroy", "Carlton",
    "Southbank", "Docklands", "Brunswick", "Hawthorn", "St Kilda",
    "Collingwood", "Prahran", "Toorak", "North Melbourne", "Footscray",
]

CERTIFICATIONS = {
    "high": "NASM CPT, ACE Certified, CrossFit Level 2, First Aid",
    "mid" : "NASM CPT, Certificate IV in Fitness, First Aid",
    "low" : "Certificate III in Fitness, First Aid",
}

EDUCATION = {
    "high": "Bachelor of Exercise Science",
    "mid" : "Diploma of Fitness",
    "low" : "Certificate IV in Fitness",
}

AVAILABILITY = {
    "high": "Mon-Sat mornings and evenings",
    "mid" : "Mon-Fri mornings, Wed-Fri evenings",
    "low" : "Tue-Thu mornings, Sat mornings only",
}


def tier(score):
    if score >= 0.65:
        return "high"
    elif score >= 0.40:
        return "mid"
    return "low"


def gen_age_range(avg_age_score):
    if avg_age_score >= 0.65:
        return 18, 65
    elif avg_age_score >= 0.40:
        lo = random.randint(20, 30)
        return lo, lo + random.randint(20, 25)
    lo = random.randint(25, 35)
    return lo, lo + random.randint(10, 15)


def gen_experience(avg_exp_score):
    if avg_exp_score >= 0.65:
        return random.randint(8, 15)
    elif avg_exp_score >= 0.40:
        return random.randint(3, 7)
    return random.randint(1, 2)


def gen_last_updated(avg_recency_score):
    if avg_recency_score >= 0.65:
        days_ago = random.randint(1, 30)
    elif avg_recency_score >= 0.40:
        days_ago = random.randint(31, 180)
    else:
        days_ago = random.randint(181, 730)
    return (datetime(2026, 6, 1) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def gen_gender(avg_gender_score):
    if avg_gender_score >= 0.70:
        return "Any"
    return random.choice(["Male", "Female"])


def gen_location(avg_location_score):
    if avg_location_score >= 0.65:
        return random.choice(LOCATIONS[:5])
    elif avg_location_score >= 0.40:
        return random.choice(LOCATIONS[5:10])
    return random.choice(LOCATIONS[10:])


# load source data
print("Loading source data...")
trainers = pd.read_excel(EXCEL_PATH, sheet_name="trainers")
df_csv   = pd.read_csv(CSV_PATH)

factor_cols = [
    "gender_score", "availability_score", "age_score",
    "qualification_score", "education_score", "location_score",
    "recency_score", "experience_score",
]

# compute per-trainer averages from the 1000-pair CSV
avg = df_csv.groupby("trainer_id")[factor_cols].mean().reset_index()
avg["excel_id"] = avg["trainer_id"].apply(
    lambda x: "T" + x.split("-")[1][-3:] if x.startswith("TRN-") else x
)
avg = avg.set_index("excel_id")

# build enriched profiles
rows = []
for _, t in trainers.iterrows():
    tid  = str(t["trainer_id"])
    avgs = avg.loc[tid] if tid in avg.index else pd.Series({c: 0.5 for c in factor_cols})

    age_min, age_max = gen_age_range(avgs["age_score"])

    rows.append({
        # text factors (from Excel)
        "trainer_id"         : tid,
        "goal_tags"          : str(t.get("goal_tags", "")),
        "styles_offered"     : str(t.get("styles_offered", "")),
        "personality_traits" : str(t.get("personality_traits", "")),
        "bio"                : str(t.get("bio", "")),
        # raw data for the 8 non-text factors
        "gender"             : gen_gender(avgs["gender_score"]),
        "availability"       : AVAILABILITY[tier(avgs["availability_score"])],
        "age_range_min"      : age_min,
        "age_range_max"      : age_max,
        "certifications"     : CERTIFICATIONS[tier(avgs["qualification_score"])],
        "education"          : EDUCATION[tier(avgs["education_score"])],
        "location"           : gen_location(avgs["location_score"]),
        "last_updated"       : gen_last_updated(avgs["recency_score"]),
        "years_experience"   : gen_experience(avgs["experience_score"]),
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_PATH, index=False)

print(f"\nGenerated {len(result)} trainer profiles -> {OUTPUT_PATH}")
print("\nSample:")
print(result[["trainer_id", "gender", "location", "years_experience",
              "age_range_min", "age_range_max", "education"]].to_string())
