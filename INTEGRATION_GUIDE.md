# VTC AI Matching — Integration Guide

RMIT Industry Project 2026 - For the VTC engineering team

---

The matching edge function already scores trainers across 11 factors. This integration improves three of them. Instead of exact keyword matching (Jaccard) for goals, training style, and personality, a Python service uses sentence embeddings that understand meaning. "Fat loss" and "body recomposition" mean the same thing - Jaccard scores them as zero, embeddings score them at ~0.89.

The other 8 factors (gender, availability, age, qualifications, education, location, recency, experience) are untouched. The weighted average formula in TypeScript stays exactly as it is. You're only changing the source of three values going into it.

**What changes:**

| | Before | After |
|---|---|---|
| Goals, Style, Persona scoring | Jaccard keyword overlap | Semantic embedding similarity |
| Factor weights | Hardcoded defaults | Data-derived from synthetic benchmark |
| Match card explanation | Raw subscore JSON | Plain-English string |

---

## How it fits together

```
Client request
      │
      ▼
match-candidates/index.ts  (unchanged structure)
      │
      ├─► Hard filters, gender filter, approval - UNCHANGED
      │
      ├─► Availability, location, age, quals, etc. - UNCHANGED
      │
      └─► POST /batch-score  →  Python AI service (deployed separately)
                                    Returns goal_score, style_score, persona_score
                │
                ▼
          Slot those 3 scores into the existing 11-factor weighted total
                │
                ▼
          POST /explain  →  plain-English explanation for top matches
```

The Python service runs as a separate HTTP process — it can't run inside Supabase Edge Functions because the embedding model and numpy aren't compatible with Deno. Any platform that hosts a persistent Docker container works (Railway, Render, Fly.io, Cloud Run).

---

## The important bit: how the scores merge back in

Before making any changes to the edge function, it's worth being precise about what the Python service actually does.

The edge function computes a weighted average across all 11 factors:

```
final_score = (w₁×score₁ + w₂×score₂ + ... + w₁₁×score₁₁) / (w₁ + ... + w₁₁)
```

The Python service replaces exactly three of those inputs — the ones currently computed with Jaccard:

| Factor | Currently | After integration |
|---|---|---|
| `goal_score` | `jaccardSimilarity(clientGoals, trainer.goal_tags)` | returned by `/batch-score` |
| `style_score` | `jaccardSimilarity(clientStyles, trainer.styles_offered)` | returned by `/batch-score` |
| `persona_score` | `jaccardSimilarity(clientPersona, trainer.personality_traits)` | returned by `/batch-score` |

Everything else — the other 8 scores, the weight lookups, the weighted average itself — stays in TypeScript and doesn't change. You're swapping three inputs, not rewriting the algorithm.

---

## The API

The Python service needs to expose these four endpoints. The shapes below are the contract — the implementation is up to you.

### POST /batch-score

This is the main one to use in production. Pass the client profile and all candidate trainers in one request. It computes the client's embeddings once and reuses them across every trainer, which is much faster than calling `/semantic-score` in a loop.

```json
// Request
{
  "client": {
    "goal_tags": "weight loss, get lean",
    "styles_desired": "HIIT, strength training",
    "personality_traits": "motivating, energetic, accountability"
  },
  "trainers": [
    {
      "trainer_id": "T001",
      "goal_tags": "fat loss, body recomposition",
      "styles_offered": "interval conditioning, metabolic circuits",
      "personality_traits": "high-energy, results-driven",
      "bio": "I coach busy adults who want fat loss without fad diets."
    }
  ]
}

// Response — sorted by final_score descending
[
  {
    "trainer_id": "T001",
    "goal_score": 0.8912,
    "style_score": 0.7643,
    "persona_score": 0.7201,
    "final_score": 0.7952
  }
]
```

### POST /semantic-score

Same as above but for a single pair. Useful for testing or spot-checking a specific match — don't use this in the scoring loop.

### POST /explain

Takes a trainer profile and all 11 factor scores, returns a plain-English explanation string. If you don't include `final_score` in the scores object, the service computes it from the optimised weights automatically.

```json
// Request
{
  "trainer": {
    "trainer_id": "T001",
    "goal_tags": "fat loss, body recomposition",
    "styles_offered": "interval conditioning, metabolic circuits",
    "personality_traits": "high-energy, results-driven",
    "bio": "I coach busy adults who want fat loss without fad diets."
  },
  "scores": {
    "goal_score": 0.8912,
    "style_score": 0.7643,
    "persona_score": 0.7201,
    "gender_score": 1.0,
    "availability_score": 0.75,
    "age_score": 1.0,
    "qualification_score": 0.8,
    "education_score": 0.5,
    "location_score": 0.6,
    "recency_score": 0.9,
    "experience_score": 0.85
  }
}

// Response
{
  "explanation": "Strong match. 7 out of 11 factors are a strong match. This trainer stands out because your fitness goals align closely with their specialisation in fat loss, body recomposition, and your preferred training style matches well with what they offer (interval conditioning, metabolic circuits), and the trainer's coaching personality suits your stated preferences. Worth noting: the trainer's education level may not align with your preference."
}
```

### GET /weights

Returns the current optimised weight profile from `outputs/optimised_weight_profile.json`. Handy for inspection before importing into Supabase, or for wiring into an admin UI.

---

## Changes to match-candidates/index.ts

Read the file before making any of these changes — the variable names below are illustrative. Substitute whatever your codebase actually calls things.

### 1. Add the environment variable

In your Supabase project settings:
```
AI_SERVICE_URL=https://your-deployed-service.com
```

Locally in `.env`:
```
AI_SERVICE_URL=http://localhost:8000
```

### 2. Add a helper near the top of the edge function

```typescript
const AI_SERVICE_URL = Deno.env.get("AI_SERVICE_URL") ?? "http://localhost:8000";

interface ClientSemanticProfile {
  goal_tags: string;
  styles_desired: string;
  personality_traits: string;
}

interface TrainerSemanticProfile {
  trainer_id: string;
  goal_tags: string;
  styles_offered: string;
  personality_traits: string;
  bio: string;
}

interface SemanticScore {
  trainer_id: string;
  goal_score: number;
  style_score: number;
  persona_score: number;
  final_score: number;
}

async function getSemanticScores(
  client: ClientSemanticProfile,
  trainers: TrainerSemanticProfile[]
): Promise<Map<string, SemanticScore>> {
  try {
    const res = await fetch(`${AI_SERVICE_URL}/batch-score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client, trainers }),
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const results: SemanticScore[] = await res.json();
    return new Map(results.map((r) => [r.trainer_id, r]));
  } catch (err) {
    // Service is down — returning an empty map triggers the Jaccard fallback below
    console.error("AI service unavailable, falling back to Jaccard:", err);
    return new Map();
  }
}
```

### 3. Call it once, before the trainer-scoring loop

```typescript
// Call this once before you start iterating over candidate trainers.
// Substitute your actual variable names for yourClient and yourCandidateTrainers.
const semanticScores = await getSemanticScores(
  {
    goal_tags:          yourClient.goal_tags          ?? "",
    styles_desired:     yourClient.styles_desired     ?? "",
    personality_traits: yourClient.personality_traits ?? "",
  },
  yourCandidateTrainers.map((t) => ({
    trainer_id:         t.trainer_id,
    goal_tags:          t.goal_tags          ?? "",
    styles_offered:     t.styles_offered     ?? "",
    personality_traits: t.personality_traits ?? "",
    bio:                t.bio                ?? "",
  }))
);
```

### 4. Replace the Jaccard lines inside the scoring loop

Find the three lines where Jaccard (or equivalent) is called for goals, style, and personality, and swap them out:

```typescript
// Before
const goalScore    = jaccardSimilarity(clientGoals,   trainer.goal_tags);
const styleScore   = jaccardSimilarity(clientStyles,  trainer.styles_offered);
const personaScore = jaccardSimilarity(clientPersona, trainer.personality_traits);

// After — falls back to Jaccard automatically if the service was unreachable
const sem          = semanticScores.get(trainer.trainer_id);
const goalScore    = sem?.goal_score    ?? jaccardSimilarity(clientGoals,   trainer.goal_tags);
const styleScore   = sem?.style_score   ?? jaccardSimilarity(clientStyles,  trainer.styles_offered);
const personaScore = sem?.persona_score ?? jaccardSimilarity(clientPersona, trainer.personality_traits);

// Leave everything else in the loop as-is.
// The weighted average call doesn't change.
```

### 5. Add explanations to the top matches

After ranking, call `/explain` for however many results you're showing. At this point you already have all 11 factor scores on each trainer object — just pass them through:

```typescript
async function getExplanation(
  trainer: TrainerSemanticProfile,
  scores: Record<string, number>
): Promise<string> {
  try {
    const res = await fetch(`${AI_SERVICE_URL}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trainer, scores }),
    });
    if (!res.ok) return "";
    return (await res.json()).explanation ?? "";
  } catch {
    return "";
  }
}

const topMatchesWithExplanations = await Promise.all(
  yourRankedTrainers.slice(0, 3).map(async (match) => ({
    ...match,
    explanation: await getExplanation(
      {
        trainer_id:         match.trainer_id,
        goal_tags:          match.goal_tags          ?? "",
        styles_offered:     match.styles_offered     ?? "",
        personality_traits: match.personality_traits ?? "",
        bio:                match.bio                ?? "",
      },
      {
        goal_score:          match.goal_score,
        style_score:         match.style_score,
        persona_score:       match.persona_score,
        gender_score:        match.gender_score,
        availability_score:  match.availability_score,
        age_score:           match.age_score,
        qualification_score: match.qualification_score,
        education_score:     match.education_score,
        location_score:      match.location_score,
        recency_score:       match.recency_score,
        experience_score:    match.experience_score,
      }
    ),
  }))
);
```

Wire `explanation` into the response payload and surface it in `TrainerMatchCard` and the subscore banner in `TrainerProfileView`.

---

## Importing the optimised weights

The file `outputs/optimised_weight_profile.json` has weights derived from the synthetic training dataset. They outperform the hardcoded defaults — F1 improved from 0.574 to 0.706, accuracy from 72.5% to 87.5% on the validation set.

The biggest shifts worth knowing about: availability and qualifications ended up much more predictive than their default weights suggest (both jumped to ~4.5). Education dropped to nearly zero. Personality ended up more important than the default 0.7 assumed.

| Key | Default | Optimised |
|---|---|---|
| `w_avail` | 1.0 | 4.57 |
| `w_quals` | 1.2 | 4.50 |
| `w_persona` | 0.7 | 3.75 |
| `w_exp` | 1.0 | 3.25 |
| `w_recency` | 1.0 | 2.60 |
| `w_style` | 1.5 | 2.99 |
| `w_goals` | 1.5 | 2.46 |
| `w_loc` | 1.0 | 1.84 |
| `w_age` | 0.8 | 1.62 |
| `w_gender` | 1.0 | 1.57 |
| `w_edu` | 0.5 | 0.10 |

To apply them, run this in the Supabase SQL editor — no code changes needed:

```sql
UPDATE weight_profiles
SET
  w_gender  = 1.5662,
  w_style   = 2.9854,
  w_avail   = 4.5732,
  w_age     = 1.6212,
  w_quals   = 4.5030,
  w_edu     = 0.0978,
  w_loc     = 1.8387,
  w_recency = 2.6033,
  w_exp     = 3.2474,
  w_persona = 3.7526,
  w_goals   = 2.4569
WHERE profile_name = 'default';
```

The `decision_threshold` value in the JSON (0.6123) isn't a column in `weight_profiles` — it was used during optimisation to classify pairs as successful or not. You only need it if you build a binary gate on top of the score, which isn't part of the current algorithm.

---

## Suggested rollout order

You don't have to do all of this at once. Each step is independent and reversible.

1. **Import the weights first.** One SQL statement, no code changes, no deployment needed. You get improved ranking immediately.

2. **Add semantic scoring for goals only.** Deploy the Python service, replace just the `goal_score` Jaccard line, keep Jaccard for style and persona. Watch the match quality for a week before going further.

3. **Extend to style and persona.** Replace the other two Jaccard lines. Jaccard is now fully replaced for those three factors.

4. **Wire in explainability.** Add the `/explain` calls and surface the string in the UI.

5. **Activate the feedback loop after launch** (see below).

---

## After launch: re-training weights on real data

Once you have enough real behavioral data in `match_events`, you can re-derive the weights from actual user behaviour instead of the synthetic dataset.

Wait until you have at least ~500 confirmed booking events before running the first live update — below that the sample is too small to be reliable.

To define a "successful match" from your event log:

```sql
-- Adjust event_type values to match what your match_events table actually logs
SELECT
  client_id,
  trainer_id,
  MAX(CASE WHEN event_type = 'booking_confirmed' THEN 1 ELSE 0 END) AS match_outcome
FROM match_events
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY client_id, trainer_id;
```

Export that result as a CSV with the same column structure as `Client-Trainer_Match_Result.csv` (11 factor score columns + `match_outcome`), update `DATA_PATH` in `weight_optimisation.py` to point at it, and run:

```bash
python weight_optimisation.py
```

That produces a new `outputs/optimised_weight_profile.json`. Review it, then apply with the same SQL UPDATE as above.

The full automated pipeline design for this is in the Feedback Loop Architecture deliverable (D4).

---

## Deploying the Python service

The service needs to run as a persistent process — not a serverless function, because the embedding model (~90 MB) needs to stay loaded in memory between requests. Cold-starting it on every request would be too slow.

Recommended setup: FastAPI + uvicorn, single worker.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt fastapi uvicorn

# Download the model at build time so it's not fetched on first request
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

Single worker is correct here — multiple workers would each load a separate model copy into memory. Add a `GET /health` endpoint returning `{"status": "ok"}` so your platform can do liveness checks.

Railway, Render, Fly.io, and GCP Cloud Run all work fine for this. Any platform that runs a persistent container does.

One privacy note: the model runs entirely inside your deployment. No client or trainer data is sent to any external service — not during scoring, not during explanation generation.

---

## Module reference

| File | What it does |
|---|---|
| `jaccard_baseline.py` | Python replica of the current Jaccard scoring across all 11 factors (D1) |
| `semantic.py` | Batch semantic matching — encodes and scores all client–trainer pairs (D2) |
| `weight_optimisation.py` | Derives optimised weights from labelled data using differential evolution (D3) |
| `explainability.py` | Generates plain-English match explanations from all 11 factor scores (D5) |
| `app.py` | Streamlit demo dashboard comparing Jaccard vs semantic results |

---

*VTC x RMIT Industry Project 2026 — Confidential*
