import streamlit as st
import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Explainability_layer.explainability import explain_match

st.set_page_config(page_title="VTC Matching Dashboard", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f7f3ff; }
.title { text-align: center; color: #2c2c54; }
.card {
    background: #FBF7EF;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #E8D9BC;
    margin-bottom: 12px;
    color: #333;
}
.card-semantic {
    background: #EEF3FD;
    border-color: #A8C0E8;
}
.rank-badge {
    display: inline-block;
    background-color: #2c2c54;
    color: white;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 13px;
    margin-bottom: 6px;
}
.stat-box {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #ddd;
    text-align: center;
}
.why-box {
    background: #eef4ff;
    border-left: 4px solid #4a90d9;
    padding: 8px 12px;
    border-radius: 4px;
    margin-top: 8px;
    font-size: 13px;
    color: #2c2c54;
}
.no-match-box {
    background: #fff8f0;
    border-left: 4px solid #e67e22;
    padding: 12px 16px;
    border-radius: 6px;
    margin-top: 8px;
    color: #333;
}
.stTabs [data-baseweb="tab"] { color: #888; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #2c2c54 !important; border-bottom: 2px solid #2c2c54 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2c2c54 !important; }
div[data-testid="stMarkdownContainer"] a { color: inherit !important; text-decoration: none !important; }
div[data-testid="stMarkdownContainer"] *:hover { color: inherit !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    trainers = pd.read_excel("Semantic_matching/semantic_matching__client_trainer_dataset.xlsx", sheet_name="trainers")
    metrics = pd.read_csv("outputs/weight_optimisation_metrics.csv")
    with open("outputs/optimised_weight_profile.json") as f:
        weights = json.load(f)
    return trainers, metrics, weights

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

trainers, metrics, weights = load_data()
model = load_model()


st.markdown('<div class="title"><h1>Voltage Training Club</h1><h3>AI Trainer Matching Prototype</h3></div>', unsafe_allow_html=True)
st.write("Enter your preferences in the sidebar and click Find My Trainer to compare the current keyword-based system with the proposed semantic AI model.")
st.markdown("---")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="stat-box"><h2 style="color:#c0392b;margin:0">81.1%</h2><p style="color:#555;margin:4px 0">of Jaccard pairs scored zero<br>due to keyword mismatch</p></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-box"><h2 style="color:#27ae60;margin:0">759</h2><p style="color:#555;margin:4px 0">blind-spot pairs recovered<br>by the semantic model</p></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-box"><h2 style="color:#2c2c54;margin:0">70%</h2><p style="color:#555;margin:4px 0">of clients received a different<br>top-ranked trainer</p></div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["Trainer Recommendations", "Weight Optimisation Results"])

with tab1:

    st.sidebar.markdown("### Your Preferences")
    goal_input = st.sidebar.text_input("Your Goals", placeholder="e.g. weight loss, fat loss")
    style_input = st.sidebar.text_input("Preferred Training Styles", placeholder="e.g. HIIT, strength training")
    persona_input = st.sidebar.text_input("Personality Traits", placeholder="e.g. motivating, energetic")
    run = st.sidebar.button("Find My Trainer", use_container_width=True)

    st.sidebar.markdown("---")
    if goal_input or style_input or persona_input:
        st.sidebar.markdown("**Your Profile**")
        if goal_input:
            st.sidebar.write("Goals:", goal_input)
        if style_input:
            st.sidebar.write("Styles:", style_input)
        if persona_input:
            st.sidebar.write("Personality:", persona_input)

    if not run:
        st.info("Enter your preferences in the sidebar and click Find My Trainer to get your recommendations.")

    if run:
        if not goal_input and not style_input and not persona_input:
            st.warning("Please enter at least one preference before searching.")
        else:
            st.subheader("Your Trainer Recommendations")

            c1, c2, c3 = st.columns(3)
            c1.write("**Goals**"); c1.write(goal_input or "Not specified")
            c2.write("**Styles**"); c2.write(style_input or "Not specified")
            c3.write("**Personality**"); c3.write(persona_input or "Not specified")

            st.markdown("---")

            with st.spinner("Finding your best trainer matches..."):

                def split_tags(text):
                    if not text or pd.isna(text):
                        return set()
                    return set(t.strip().lower() for t in str(text).split(",") if t.strip())

                def jaccard(a_text, b_text):
                    a = split_tags(a_text)
                    b = split_tags(b_text)
                    if not a or not b:
                        return 0.0
                    return len(a & b) / len(a | b)

                w = {"goal": 1.5, "style": 1.5, "persona": 0.7}

                jaccard_results = []
                for _, trainer in trainers.iterrows():
                    g = round(jaccard(goal_input, trainer["goal_tags"]), 3)
                    s = round(jaccard(style_input, trainer["styles_offered"]), 3)
                    p = round(jaccard(persona_input, trainer["personality_traits"]), 3)
                    score = round((w["goal"] * g + w["style"] * s + w["persona"] * p) / sum(w.values()), 3)
                    jaccard_results.append({
                        "trainer_id": trainer["trainer_id"],
                        "goal_score": g,
                        "style_score": s,
                        "persona_score": p,
                        "final_score": score,
                    })

                jaccard_all = pd.DataFrame(jaccard_results).sort_values("final_score", ascending=False)
                jaccard_top = jaccard_all[jaccard_all["final_score"] >= 0.1].head(3).reset_index(drop=True)
                jaccard_no_match = len(jaccard_top) == 0
                if not jaccard_no_match:
                    jaccard_top["rank"] = jaccard_top.index + 1

                goal_emb = model.encode([goal_input or ""])
                style_emb = model.encode([style_input or ""])
                persona_emb = model.encode([persona_input or ""])

                trainer_goals = model.encode(trainers["goal_tags"].fillna("").tolist())
                trainer_styles = model.encode(trainers["styles_offered"].fillna("").tolist())
                trainer_personas = model.encode(trainers["personality_traits"].fillna("").tolist())
                trainer_bios = model.encode(trainers["bio"].fillna("").tolist())

                goal_scores = np.clip(cosine_similarity(goal_emb, trainer_goals)[0], 0, 1)
                style_scores = np.clip(cosine_similarity(style_emb, trainer_styles)[0], 0, 1)
                persona_trait_scores = np.clip(cosine_similarity(persona_emb, trainer_personas)[0], 0, 1)
                persona_bio_scores = np.clip(cosine_similarity(persona_emb, trainer_bios)[0], 0, 1)
                persona_scores = 0.7 * persona_trait_scores + 0.3 * persona_bio_scores
                final_scores = (1.5 * goal_scores + 1.5 * style_scores + 0.7 * persona_scores) / (1.5 + 1.5 + 0.7)

                top3_idx = np.argsort(final_scores)[::-1][:3]
                semantic_df = trainers.iloc[top3_idx].copy().reset_index(drop=True)
                semantic_df["goal_score"] = [round(float(x), 3) for x in goal_scores[top3_idx]]
                semantic_df["style_score"] = [round(float(x), 3) for x in style_scores[top3_idx]]
                semantic_df["persona_score"] = [round(float(x), 3) for x in persona_scores[top3_idx]]
                semantic_df["final_score"] = [round(float(x), 3) for x in final_scores[top3_idx]]
                semantic_df["rank"] = semantic_df.index + 1

            rank_labels = {1: "Rank 1", 2: "Rank 2", 3: "Rank 3"}

            def generate_why(goal_score, style_score, persona_score, final_score, trainer_row):
                # dashboard only computes 3 live scores; others set to 0.40 (silent zone)
                scores = {
                    "goal_score"          : goal_score,
                    "style_score"         : style_score,
                    "persona_score"       : persona_score,
                    "gender_score"        : 0.40,
                    "availability_score"  : 0.40,
                    "age_score"           : 0.40,
                    "qualification_score" : 0.40,
                    "education_score"     : 0.40,
                    "location_score"      : 0.40,
                    "recency_score"       : 0.40,
                    "experience_score"    : 0.40,
                    "final_score"         : final_score,
                }
                trainer = {
                    "goal_tags"     : str(trainer_row.get("goal_tags", "")),
                    "styles_offered": str(trainer_row.get("styles_offered", "")),
                }
                return explain_match(trainer, scores)

            left, right = st.columns(2)

            with left:
                st.subheader("Baseline Model - Jaccard")
                if jaccard_no_match:
                    st.markdown("""
                    <div class="no-match-box">
                        <b>No matches found.</b><br><br>
                        The current keyword-based system could not find any trainers matching your input.
                        This is because it only recognises exact word matches, if your words do not
                        appear verbatim in a trainer's profile, the score is zero.<br><br>
                        This is the core limitation our AI model is designed to solve.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    for _, row in jaccard_top.iterrows():
                        trainer = trainers[trainers["trainer_id"] == row["trainer_id"]].iloc[0]
                        bio = str(trainer["bio"]) if pd.notna(trainer["bio"]) else "No bio available"
                        st.markdown(f"""
                        <div class="card">
                            <span class="rank-badge">{rank_labels.get(int(row['rank']), f"Rank {int(row['rank'])}")}</span>
                            <h4 style="margin:4px 0;color:#2c2c54">{trainer['trainer_id']}</h4>
                            <p style="margin:4px 0;color:#333"><b>Specialisation:</b> {trainer['goal_tags']}</p>
                            <p style="margin:4px 0;color:#333"><b>Styles:</b> {trainer['styles_offered']}</p>
                            <p style="margin:4px 0;color:#333"><b>Bio:</b> {bio[:120]}...</p>
                            <hr style="margin:8px 0;border-color:#ddd">
                            <p style="margin:2px 0;color:#333">Goal: <b>{row['goal_score']}</b> &nbsp;|&nbsp; Style: <b>{row['style_score']}</b> &nbsp;|&nbsp; Persona: <b>{row['persona_score']}</b></p>
                            <h4 style="margin:6px 0;color:#2c2c54">Overall Score: {row['final_score']}</h4>
                        </div>
                        """, unsafe_allow_html=True)

            with right:
                st.subheader("Semantic AI Model")
                for _, row in semantic_df.iterrows():
                    bio = str(row["bio"]) if pd.notna(row["bio"]) else "No bio available"
                    why = generate_why(row["goal_score"], row["style_score"], row["persona_score"], row["final_score"], row)
                    st.markdown(f"""
                    <div class="card card-semantic">
                        <span class="rank-badge">{rank_labels.get(int(row['rank']), f"Rank {int(row['rank'])}")}</span>
                        <h4 style="margin:4px 0;color:#2c2c54">{row['trainer_id']}</h4>
                        <p style="margin:4px 0;color:#333"><b>Specialisation:</b> {row['goal_tags']}</p>
                        <p style="margin:4px 0;color:#333"><b>Styles:</b> {row['styles_offered']}</p>
                        <p style="margin:4px 0;color:#333"><b>Bio:</b> {bio[:120]}...</p>
                        <hr style="margin:8px 0;border-color:#ddd">
                        <p style="margin:2px 0;color:#333">Goal: <b>{row['goal_score']}</b> &nbsp;|&nbsp; Style: <b>{row['style_score']}</b> &nbsp;|&nbsp; Persona: <b>{row['persona_score']}</b></p>
                        <h4 style="margin:6px 0;color:#2c2c54">Overall Score: {row['final_score']}</h4>
                        <div class="why-box">{why}</div>
                    </div>
                    """, unsafe_allow_html=True)


with tab2:

    st.subheader("Model Performance Metrics")
    st.write("The table below shows how the optimised weights compare against the default hardcoded weights on the validation set.")
    metrics_display = metrics.drop(columns=["threshold"])
    st.table(metrics_display)

    st.markdown("---")

    st.subheader("Optimised Weight Profile")
    st.write("These weights were learned from the synthetic labelled dataset using differential evolution optimisation via scipy. They replace the original hardcoded values and can be imported directly into VTC's weight profiles table.")

    default_weights = {
        "w_gender": 1.0, "w_style": 1.5, "w_avail": 1.0, "w_age": 0.8,
        "w_quals": 1.2, "w_edu": 0.5, "w_loc": 1.0, "w_recency": 1.0,
        "w_exp": 1.0, "w_persona": 0.7, "w_goals": 1.5
    }

    filtered_weights = {k: v for k, v in weights.items() if k != "decision_threshold"}
    weight_table = pd.DataFrame({
        "Factor": list(filtered_weights.keys()),
        "Default Weight": [default_weights[k] for k in filtered_weights.keys()],
        "Optimised Weight": list(filtered_weights.values())
    })
    st.table(weight_table)

    st.markdown("---")

    st.subheader("Prototype Note")
    st.write(
        "This is a proof-of-concept prototype built on a synthetic dataset. "
        "It demonstrates how semantic AI matching improves trainer recommendations "
        "compared to exact keyword matching, and how data-driven weight optimisation "
        "can replace hardcoded factor weights. The system is designed to improve "
        "further once real post-launch user data becomes available."
    )
