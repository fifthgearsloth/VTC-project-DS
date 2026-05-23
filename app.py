import streamlit as st
import pandas as pd
import json

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

.stTabs [data-baseweb="tab"] { color: #888; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #2c2c54 !important; border-bottom: 2px solid #2c2c54 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #2c2c54 !important; }

.stSelectbox [data-baseweb="select"] { cursor: pointer !important; }
.stSelectbox [data-baseweb="select"] * { cursor: pointer !important; }
.stSelectbox [data-baseweb="select"] > div { border-color: #2c2c54 !important; }

div[data-testid="stMarkdownContainer"] a { color: inherit !important; text-decoration: none !important; }
div[data-testid="stMarkdownContainer"] *:hover { color: inherit !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    clients = pd.read_excel("semantic_matching__client_trainer_dataset.xlsx", sheet_name="clients")
    trainers = pd.read_excel("semantic_matching__client_trainer_dataset.xlsx", sheet_name="trainers")
    jaccard = pd.read_csv("jaccard_full_results.csv")
    semantic = pd.read_csv("semantic_full_results.csv")
    metrics = pd.read_csv("outputs/weight_optimisation_metrics.csv")
    with open("outputs/optimised_weight_profile.json") as f:
        weights = json.load(f)
    return clients, trainers, jaccard, semantic, metrics, weights

clients, trainers, jaccard, semantic, metrics, weights = load_data()


st.markdown('<div class="title"><h1>Voltage Training Club</h1><h3>AI Trainer Matching Prototype</h3></div>', unsafe_allow_html=True)
st.write("This dashboard compares the current keyword-based matching system with the proposed semantic AI matching model.")
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

    client_id = st.sidebar.selectbox("Select Client", clients["client_id"])
    client = clients[clients["client_id"] == client_id].iloc[0]

    st.sidebar.subheader("Client Profile")
    st.sidebar.write("**Goals:**", client["goal_tags"])
    st.sidebar.write("**Styles:**", client["styles_desired"])
    st.sidebar.write("**Personality:**", client["personality_traits"])

    st.subheader(f"Recommendations for Client {client_id}")

    c1, c2, c3 = st.columns(3)
    c1.write("**Goals**"); c1.write(client["goal_tags"])
    c2.write("**Styles**"); c2.write(client["styles_desired"])
    c3.write("**Personality**"); c3.write(client["personality_traits"])

    st.markdown("---")

    rank_labels = {1: "Rank 1", 2: "Rank 2", 3: "Rank 3"}

    def generate_why(client, trainer, row):
        reasons = []
        if row["goal_score"] >= 0.5:
            reasons.append(f"your goals align closely with this trainer's specialisation in {trainer['goal_tags'][:40]}")
        if row["style_score"] >= 0.5:
            reasons.append("your preferred training styles match well with what this trainer offers")
        if row["persona_score"] >= 0.5:
            reasons.append("the trainer's personality and coaching approach suit your stated preferences")
        if not reasons:
            reasons.append("this trainer is the closest available match based on your overall profile")
        return "Recommended because " + ", and ".join(reasons) + "."

    def show_results(title, results, card_class, is_semantic=False):
        st.subheader(title)
        top3 = results[results["client_id"] == client_id].nsmallest(3, "rank")
        for _, row in top3.iterrows():
            trainer = trainers[trainers["trainer_id"] == row["trainer_id"]].iloc[0]
            bio = str(trainer["bio"]) if pd.notna(trainer["bio"]) else "No bio available"
            why_html = f'<div class="why-box">{generate_why(client, trainer, row)}</div>' if is_semantic else ""
            st.markdown(f"""
            <div class="card {card_class}">
                <span class="rank-badge">{rank_labels.get(int(row["rank"]), f"Rank {int(row['rank'])}")}</span>
                <h4 style="margin:4px 0;color:#2c2c54">{trainer['trainer_id']}</h4>
                <p style="margin:4px 0;color:#333"><b>Specialisation:</b> {trainer['goal_tags']}</p>
                <p style="margin:4px 0;color:#333"><b>Styles:</b> {trainer['styles_offered']}</p>
                <p style="margin:4px 0;color:#333"><b>Bio:</b> {bio[:120]}...</p>
                <hr style="margin:8px 0;border-color:#ddd">
                <p style="margin:2px 0;color:#333">Goal: <b>{row['goal_score']}</b> &nbsp;|&nbsp; Style: <b>{row['style_score']}</b> &nbsp;|&nbsp; Persona: <b>{row['persona_score']}</b></p>
                <h4 style="margin:6px 0;color:#2c2c54">Overall Score: {row['final_score']}</h4>
                {why_html}
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        show_results("Baseline Model - Jaccard", jaccard, "", is_semantic=False)
    with right:
        show_results("Semantic AI Model", semantic, "card-semantic", is_semantic=True)


with tab2:

    st.subheader("Model Performance Metrics")
    st.write("The table below shows how the optimised weights compare against the default hardcoded weights on the validation set.")
    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")

    st.subheader("Optimised Weight Profile")
    st.write("These weights were learned from the synthetic labelled dataset using differential evolution optimisation via scipy. They replace the original hardcoded values and can be imported directly into VTC's weight profiles table.")

    default_weights = {
        "w_gender": 1.0, "w_style": 1.5, "w_avail": 1.0, "w_age": 0.8,
        "w_quals": 1.2, "w_edu": 0.5, "w_loc": 1.0, "w_recency": 1.0,
        "w_exp": 1.0, "w_persona": 0.7, "w_goals": 1.5
    }

    weight_table = pd.DataFrame({
        "Factor": list(weights.keys()),
        "Default Weight": [default_weights[k] for k in weights.keys()],
        "Optimised Weight": list(weights.values())
    })
    st.dataframe(weight_table, use_container_width=True)

    st.markdown("---")

    st.subheader("Prototype Note")
    st.write(
        "This is a proof-of-concept prototype built on a synthetic dataset. "
        "It demonstrates how semantic AI matching improves trainer recommendations "
        "compared to exact keyword matching, and how data-driven weight optimisation "
        "can replace hardcoded factor weights. The system is designed to improve "
        "further once real post-launch user data becomes available."
    )