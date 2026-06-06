# vtc_chatbot.py
# VTC AI Trainer Matching Chatbot
#
# A conversational chatbot that matches clients to personal trainers.
# Uses Groq (free LLM) for natural language understanding and response generation.
# Uses sentence-transformers for semantic matching and explainability.py for explanations.
#
# Setup:
#   pip install groq
#   Get a free API key at https://console.groq.com (takes 2 minutes)
#
# Run: streamlit run vtc_chatbot.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from explainability import explain_match

EXCEL_PATH = "semantic_matching__client_trainer_dataset.xlsx"
LLM_MODEL  = "llama-3.3-70b-versatile"   # free on Groq, supports tool calling

SYSTEM_PROMPT = """You are the Voltage Training Club (VTC) AI Trainer Matching Assistant.
Your job is to help clients find the right personal trainer through friendly conversation.

When a client tells you what they are looking for, use the search_trainers tool to find matches.
Never make up trainer names - always call the tool first.

After getting results, present the top trainers warmly and conversationally.
For each trainer, mention their match score and why they are a good fit.
Keep responses concise and encouraging.

If the client wants to refine their search, call the tool again with updated preferences.
If they ask general fitness questions, answer helpfully."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_trainers",
        "description": "Find the best matching trainers for a client based on their preferences. Call this whenever a client describes what they are looking for in a trainer.",
        "parameters": {
            "type": "object",
            "properties": {
                "goals": {
                    "type": "string",
                    "description": "The client's fitness goals, e.g. weight loss, muscle gain, marathon training, flexibility"
                },
                "style": {
                    "type": "string",
                    "description": "The client's preferred training style, e.g. HIIT, strength training, yoga, pilates, cardio"
                },
                "persona": {
                    "type": "string",
                    "description": "The type of trainer personality the client prefers, e.g. motivating, calm, energetic, strict, supportive"
                },
            },
            "required": ["goals"],
        },
    },
}]


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_trainers():
    return pd.read_excel(EXCEL_PATH, sheet_name="trainers")

@st.cache_resource
def encode_trainers(_trainers):
    """Pre-encode all trainer profiles so matching is fast."""
    mdl = load_model()
    return (
        mdl.encode(_trainers["goal_tags"].fillna("").tolist()),
        mdl.encode(_trainers["styles_offered"].fillna("").tolist()),
        mdl.encode(_trainers["personality_traits"].fillna("").tolist()),
        mdl.encode(_trainers["bio"].fillna("").tolist()),
    )


def search_trainers(goals: str, style: str = "", persona: str = "") -> dict:
    """
    Semantic matching pipeline.
    Returns top 3 trainers with match scores and plain English explanations.
    """
    mdl      = load_model()
    trainers = load_trainers()
    t_goals, t_styles, t_personas, t_bios = encode_trainers(trainers)

    g_emb = mdl.encode([goals])
    s_emb = mdl.encode([style or ""])
    p_emb = mdl.encode([persona or ""])

    gs = np.clip(cosine_similarity(g_emb, t_goals)[0], 0, 1)
    ss = np.clip(cosine_similarity(s_emb, t_styles)[0], 0, 1)
    ps = np.clip(
        0.7 * cosine_similarity(p_emb, t_personas)[0] +
        0.3 * cosine_similarity(p_emb, t_bios)[0], 0, 1
    )
    fs = (1.5 * gs + 1.5 * ss + 0.7 * ps) / (1.5 + 1.5 + 0.7)

    top3 = np.argsort(fs)[::-1][:3]
    results = []

    for rank, idx in enumerate(top3, 1):
        t = trainers.iloc[idx]
        profile = {
            "trainer_id"        : str(t["trainer_id"]),
            "goal_tags"         : str(t.get("goal_tags", "")),
            "styles_offered"    : str(t.get("styles_offered", "")),
            "personality_traits": str(t.get("personality_traits", "")),
            "bio"               : str(t.get("bio", "")),
        }
        # 3 live scores from semantic matching, 8 set to neutral (0.5)
        # because gender, schedule, location etc. are not collected in this query
        scores = {
            "goal_score"          : round(float(gs[idx]), 4),
            "style_score"         : round(float(ss[idx]), 4),
            "persona_score"       : round(float(ps[idx]), 4),
            "gender_score"        : 0.5,
            "availability_score"  : 0.5,
            "age_score"           : 0.5,
            "qualification_score" : 0.5,
            "education_score"     : 0.5,
            "location_score"      : 0.5,
            "recency_score"       : 0.5,
            "experience_score"    : 0.5,
            "final_score"         : round(float(fs[idx]), 4),
        }
        explanation = explain_match(profile, scores)
        results.append({
            "rank"       : rank,
            "trainer_id" : profile["trainer_id"],
            "score"      : scores["final_score"],
            "goal_tags"  : profile["goal_tags"],
            "styles"     : profile["styles_offered"],
            "bio"        : profile["bio"][:150] + "...",
            "explanation": explanation,
        })

    return {"matches": results, "total_trainers_searched": len(trainers)}


def call_llm(messages: list, api_key: str) -> tuple:
    """
    Send messages to Groq LLM. If it calls the search_trainers tool,
    run the actual matching and send results back for a final response.
    """
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message

    # LLM wants to run trainer search
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args      = json.loads(tool_call.function.arguments)

        # run the actual matching
        result = search_trainers(**args)

        # add tool call + result to message history
        messages.append({
            "role"      : "assistant",
            "content"   : msg.content or "",
            "tool_calls": [{
                "id"      : tool_call.id,
                "type"    : "function",
                "function": {
                    "name"     : tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        })
        messages.append({
            "role"        : "tool",
            "tool_call_id": tool_call.id,
            "content"     : json.dumps(result),
        })

        # second LLM call to format the results as a conversational response
        final = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        reply = final.choices[0].message.content
    else:
        reply = msg.content

    messages.append({"role": "assistant", "content": reply})
    return reply, messages


# page setup
st.set_page_config(page_title="VTC Trainer Matching Chatbot", layout="centered")
st.markdown("""
<style>
.stApp { background-color: #f7f3ff; }
</style>
""", unsafe_allow_html=True)

st.title("Voltage Training Club")
st.subheader("AI Trainer Matching Chatbot")
st.caption("Tell me what you are looking for and I will find you the best trainer.")

# API key input in sidebar
st.sidebar.markdown("### Setup")
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="Get a free key at console.groq.com",
    help="Your key is never stored. Get a free one at https://console.groq.com"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown(
    "1. Tell the chatbot your fitness goals\n"
    "2. The AI matches you to trainers using semantic search\n"
    "3. You get a plain English explanation for each recommendation\n"
)

# initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
if prompt := st.chat_input("Tell me your fitness goals..."):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to get started. It is free at console.groq.com")
        st.stop()

    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # add to LLM history and call
    st.session_state.llm_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Finding your best trainers..."):
            try:
                reply, updated_history = call_llm(st.session_state.llm_history, api_key)
                st.session_state.llm_history = updated_history
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                err = f"Something went wrong: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
