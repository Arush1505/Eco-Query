from openai import OpenAI
import streamlit as st
import pandas as pd
import requests
import json
import ast
import time
import re
import nltk
from nltk.stem import WordNetLemmatizer
from thefuzz import fuzz
from dotenv import load_dotenv
import os
load_dotenv()

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Mindful Queries",
    page_icon="🧠",
    layout="centered"
)

# ==============================
# Configuration
# ==============================
SLM_MODEL = "qwen2:1.5b"
LLM_MODEL = "llama3.1:8b"
KB_PATH   = "../knowledge_dataset_llm.csv"

# ✏️ Paste your HuggingFace token here
# Get it from: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODELS = {
    SLM_MODEL: "Qwen/Qwen2-1.5B-Instruct:featherless-ai",
    LLM_MODEL: "meta-llama/Llama-3.1-8B-Instruct:novita"
}

ALLOWED_INTENTS = {
    "definition",
    "advantages",
    "disadvantages",
    "applications",
    "complexity",
    "code",
    "comparison",
    "others"
}

GENERIC_TOPIC_BLOCKLIST = {
    "algorithm", "algorithms",
    "data structure", "data structures",
    "programming", "string",
    "code", "concept", "method",
    "technique", "structure"
}

ATTRIBUTE_TOKENS = {
    "time complexity",
    "space complexity",
    "complexity",
    "proof",
    "implementation",
    "example",
    "applications",
    "advantages",
    "disadvantages",
    "code"
}

FUZZY_THRESHOLD = 70

# ==============================
# NLTK Init
# ==============================
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
lemmatizer = WordNetLemmatizer()


# ==============================
# Utility Functions
# ==============================
def normalize_text(text: str) -> str:
    """
    Strong normalization:
    - lowercase
    - remove hyphen
    - remove punctuation
    - collapse whitespace
    - lemmatize
    """
    text = str(text).lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return lemmatizer.lemmatize(text)


# ==============================
# Knowledge Base Loader
# ==============================
@st.cache_data
def load_kb():
    df = pd.read_csv(KB_PATH)
    df["heading_norm"] = df["heading"].apply(normalize_text)

    def parse_aliases(x):
        if pd.isna(x):
            return []
        try:
            aliases = ast.literal_eval(x)
        except:
            aliases = [x]
        return [normalize_text(a) for a in aliases]

    def parse_extras(x):
        if pd.isna(x):
            return []
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                items = v
            else:
                items = [v]
        except:
            items = [s.strip() for s in str(x).split(",") if s.strip()]
        return [normalize_text(a) for a in items]

    df["aliases_norm"] = df["aliases"].apply(parse_aliases)
    df["extras_norm"]  = df["extras"].apply(parse_extras)
    return df


# ==============================
# Fuzzy Topic Matching
# ==============================
def match_topic_fuzzy(kb_df, topic, threshold=FUZZY_THRESHOLD):
    topic_norm = normalize_text(topic)
    best_score = 0
    best_row   = None
    best_field = None

    for _, row in kb_df.iterrows():
        heading_score = fuzz.token_set_ratio(topic_norm, row["heading_norm"])

        alias_score = 0
        for alias in row["aliases_norm"]:
            alias_score = max(alias_score, fuzz.token_set_ratio(topic_norm, alias))

        extras_score = 0
        for extra in row.get("extras_norm", []):
            extras_score = max(extras_score, fuzz.token_set_ratio(topic_norm, extra))

        row_field_score = {
            "heading": heading_score,
            "alias"  : alias_score,
            "extras" : extras_score
        }
        field, score = max(row_field_score.items(), key=lambda kv: kv[1])

        if score > best_score:
            best_score = score
            best_row   = row
            best_field = field

    if best_score >= threshold:
        return {"row": best_row.to_dict(), "field": best_field, "score": best_score}
    return None


# ================================================================
# HuggingFace API Call
# (replaces call_ollama — this is the ONLY change from original)
# Function name kept identical so zero changes needed elsewhere.
# ================================================================
def call_ollama(model, prompt):
    MAX_RETRIES = 3

    model_id = HF_MODELS.get(model)
    if not model_id:
        return f"MODEL_NOT_FOUND: {model}"

    for attempt in range(MAX_RETRIES):
        try:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKEN
            )
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            if "429" in err:
                st.toast("⚠️ Rate limit hit, waiting 30s...", icon="⚠️")
                time.sleep(30)
                continue
            if "503" in err:
                st.toast("⏳ Model loading, waiting 20s...", icon="⏳")
                time.sleep(20)
                continue
            if "401" in err:
                return "HF_AUTH_ERROR: Invalid token."
            if "403" in err:
                return "HF_ACCESS_ERROR: Request model access on HuggingFace."
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
                continue
            return f"CONNECTION_ERROR: {err}"

    return "CONNECTION_ERROR: Max retries exceeded"

# ==============================
# Metadata Extraction  (unchanged from original)
# ==============================
def extract_metadata(user_prompt):

    MAX_RETRIES = 3
    attempt     = 0

    while attempt < MAX_RETRIES:

        extraction_prompt = f"""
        You are a STRICT academic query classifier and metadata extractor.
        ---------------------------------
        Topic Extraction Instruction:
        ---------------------------------
        Identify each topic AND its specific intent as PAIRS.
        
        Warning : Analyse the sentence , then Map each Topic with it's intent. Do not merge the intents.

        Generic keywords should not be included in output topic list.
        - Data Structures
        - Algorithms
        - Operating Systems
        - DBMS
        - Computer Networks
        - Theory of Computation
        - Compilers
        - Software Engineering

        Your task has TWO steps:
        1) Decide whether the query is purely academic (computer science only).
        2) If academic: extract topic-intent PAIRS. Else: is_academic=false, queries=[].

        IMPORTANT RULES:
        1) If the query contains ANY non-computer-science request
        (cooking, travel, fitness, history, sports, health, lifestyle, etc.)
        then:
            - is_academic = false
            - queries = []

        ----------------------
        Output Rules:
        ----------------------
        Return ONLY valid JSON.
        No explanations. No markdown. No comments.

        REQUIRED JSON STRUCTURE EXACTLY:
        {{
          "queries": [
            {{"topic": "", "intent": ""}},
            {{"topic": "", "intent": ""}}
          ],
          "is_academic": true/false
        }}

        Intents must come ONLY from:
        {list(ALLOWED_INTENTS)}

        -----------------------------------
        DEFINITION OF ACADEMIC QUERY:
        -----------------------------------
        The query must be ENTIRELY about Computer Science concepts.

        Examples:

        Query: What are advantages of Binary Tree. Explain the definition of Arrays.
        Output:
        {{
          "queries": [
            {{"topic": "Binary Tree", "intent": "advantages"}},
            {{"topic": "Arrays",      "intent": "definition"}}
          ],
          "is_academic": true
        }}

        Query: What are advantages of Stack and Queue?
        Output:
        {{
          "queries": [
            {{"topic": "Stack", "intent": "advantages"}},
            {{"topic": "Queue", "intent": "advantages"}}
          ],
          "is_academic": true
        }}

        Query: Explain Dijkstra's algorithm.
        Output:
        {{
          "queries": [
            {{"topic": "Dijkstra's Algorithm", "intent": "definition"}}
          ],
          "is_academic": true
        }}

        Query: Explain Dijkstra's algorithm and tell me how to clean a skillet.
        Output:
        {{
          "queries": [],
          "is_academic": false
        }}

        Query: How do I bake a cake?
        Output:
        {{
          "queries": [],
          "is_academic": false
        }}

        -----------------------------------
        Now classify the following query:
        {user_prompt}
        """

        raw_response = call_ollama(SLM_MODEL, extraction_prompt)

        # ---------------------------
        # Basic JSON Extraction
        # ---------------------------
        try:
            start = raw_response.find("{")
            end   = raw_response.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found")
            json_str = raw_response[start:end]
            parsed   = json.loads(json_str)
        except Exception:
            attempt += 1
            continue

        # ---------------------------
        # Structural Validation
        # ---------------------------
        if not isinstance(parsed, dict):
            attempt += 1
            continue

        is_academic = parsed.get("is_academic", False)
        raw_queries = parsed.get("queries", [])

        if not isinstance(is_academic, bool):
            attempt += 1
            continue

        if not isinstance(raw_queries, list):
            attempt += 1
            continue

        # Validate each topic-intent pair
        clean_pairs = []
        valid = True
        for item in raw_queries:
            if not isinstance(item, dict):
                valid = False
                break

            topic  = item.get("topic", "").strip()
            intent = item.get("intent", "").lower().strip()

            # Skip empty topics
            if not topic:
                continue

            # Skip generic topics
            if normalize_text(topic) in GENERIC_TOPIC_BLOCKLIST:
                continue

            # Fallback intent if not recognized
            if intent not in ALLOWED_INTENTS:
                intent = "definition"

            clean_pairs.append({"topic": topic, "intent": intent})

        if not valid:
            attempt += 1
            continue

        # Fallback if academic but no pairs extracted
        if is_academic and not clean_pairs:
            attempt += 1
            continue

        return {
            "queries"    : clean_pairs,
            "is_academic": is_academic
        }

    # Fallback after 3 failures
    return {"queries": [], "is_academic": False}

# ==============================
# Academic Router  (unchanged from original)
# ==============================
def academic_router(query, kb_df):

    total_start = time.perf_counter()

    # 1. Metadata Extraction
    extraction_start   = time.perf_counter()
    meta               = extract_metadata(query)
    extraction_latency = time.perf_counter() - extraction_start

    # 2. Immediate Non-Academic Redirect
    if not meta.get("is_academic", False):
        format_start   = time.perf_counter()
        response       = call_ollama(LLM_MODEL, query)
        format_latency = time.perf_counter() - format_start

        return response, "LLM", meta, {
            "extraction": extraction_latency,
            "kb_search" : 0.0,
            "formatting": format_latency,
            "total"     : time.perf_counter() - total_start
        }

    # 3. KB-Grounded Topic Validation
    kb_search_start = time.perf_counter()

    validated_pairs       = []   # {"topic", "intent", "row"}
    attribute_validations = {}
    non_kb_topics         = []

    # First pass: strong matches (heading/alias) — per pair
    for pair in meta.get("queries", []):
        topic  = pair["topic"]
        intent = pair["intent"]

        match = match_topic_fuzzy(kb_df, topic, threshold=FUZZY_THRESHOLD)
        if match and match["field"] in ("heading", "alias"):
            validated_pairs.append({
                "topic" : topic,
                "intent": intent,
                "row"   : match["row"]
            })
        else:
            non_kb_topics.append(topic)

    # Second pass: attribute-like tokens
    validated_topics = [p["topic"] for p in validated_pairs]
    validated_rows   = [p["row"]   for p in validated_pairs]

    still_unresolved = []
    for topic in non_kb_topics:
        t_norm = normalize_text(topic)
        if t_norm in ATTRIBUTE_TOKENS:
            matched_attr = False
            for row in validated_rows:
                extras = row.get("extras_norm", [])
                if t_norm in extras:
                    attribute_validations[topic] = row["heading"]
                    matched_attr = True
                    break
            if not matched_attr:
                still_unresolved.append(topic)
        else:
            still_unresolved.append(topic)

    # Third pass: lower-threshold extras-wide search
    for topic in still_unresolved:
        match = match_topic_fuzzy(kb_df, topic, threshold=60)
        if match and match["field"] == "extras":
            if validated_rows or match["score"] >= 85:
                # Use intent from original pair if available, else default
                original_intent = next(
                    (p["intent"] for p in meta.get("queries", [])
                     if p["topic"] == topic),
                    "definition"
                )
                validated_pairs.append({
                    "topic" : topic,
                    "intent": original_intent,
                    "row"   : match["row"]
                })
                validated_topics.append(topic)
                validated_rows.append(match["row"])

    all_query_topics = [p["topic"] for p in meta.get("queries", [])]
    final_non_kb = [
        t for t in all_query_topics
        if t not in validated_topics and t not in attribute_validations
    ]

    kb_search_latency = time.perf_counter() - kb_search_start

    # Case A: No KB grounding → LLM
    if len(validated_pairs) == 0:
        format_start   = time.perf_counter()
        response       = call_ollama(LLM_MODEL, query)
        format_latency = time.perf_counter() - format_start

        return response, "LLM", meta, {
            "extraction": extraction_latency,
            "kb_search" : kb_search_latency,
            "formatting": format_latency,
            "total"     : time.perf_counter() - total_start
        }

    # Case B: Mixed query → LLM
    if len(final_non_kb) > 0:
        format_start   = time.perf_counter()
        response       = call_ollama(LLM_MODEL, query)
        format_latency = time.perf_counter() - format_start

        return response, "LLM", meta, {
            "extraction": extraction_latency,
            "kb_search" : kb_search_latency,
            "formatting": format_latency,
            "total"     : time.perf_counter() - total_start
        }

    # 4. Pure Academic → SLM
    # Build per-pair targeted data — only fetch what user asked for
    field_map_keys = {
        "definition"   : "definition",
        "advantages"   : "advantages",
        "disadvantages": "disadvantages",
        "applications" : "applications",
        "code"         : "code",
        "complexity"   : "extras",
        "comparison"   : "extras",
        "others"       : "extras"
    }

    topic_intent_data = []
    for pair in validated_pairs:
        row        = pair["row"]
        intent     = pair["intent"]
        field_key  = field_map_keys.get(intent, "extras")
        data_value = row.get(field_key, "") or row.get("extras", "")

        topic_intent_data.append({
            "topic" : pair["topic"],
            "intent": intent,
            "data"  : data_value
        })

    synth_prompt = f"""
    You are an academic assistant.

    Question:
    {query}

    For each topic below, answer ONLY its paired intent using ONLY the provided data.
    Do NOT mix intents across topics.

    Topic-Intent-Data:
    {json.dumps(topic_intent_data, indent=2)}

    Attached Attributes:
    {attribute_validations}

    Instructions:
    - Answer each topic with its specific intent only
    - Do not hallucinate information not present in the data
    - Structure response clearly per topic
    - Do not add extra information beyond what is asked
    """

    format_start   = time.perf_counter()
    response       = call_ollama(SLM_MODEL, synth_prompt)
    format_latency = time.perf_counter() - format_start

    return response, "SLM", meta, {
        "extraction": extraction_latency,
        "kb_search" : kb_search_latency,
        "formatting": format_latency,
        "total"     : time.perf_counter() - total_start
    }


# ==============================
# Streamlit UI
# ==============================
st.title("🧠 Mindful Queries")
st.caption("Energy-aware academic query routing · SLM (Qwen2) + LLM (Llama 3.1) via HuggingFace")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # hf_token_input = st.text_input(
    #     "HuggingFace Token",
    #     value=HF_TOKEN,
    #     type="password",
    #     help="Get yours at https://huggingface.co/settings/tokens"
    # )
    # Allow runtime token override without redeploying
    # if hf_token_input:
    #     HF_TOKEN = hf_token_input

    st.divider()
    st.markdown("**Models**")
    st.code(f"SLM : Qwen2-1.5B-Instruct\nLLM : Llama-3.1-8B-Instruct")
    st.caption("Both served via HuggingFace Inference API — no local server needed")

    st.divider()

if st.button("🔌 Test HF Connection"):
        test_client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )

        with st.spinner("Testing SLM (Qwen2)..."):
            try:
                slm_resp = test_client.chat.completions.create(
                    model=HF_MODELS[SLM_MODEL],
                    messages=[{"role": "user", "content": "say hi"}],
                    max_tokens=5
                )
                st.success(f"✅ SLM connected! → {slm_resp.choices[0].message.content.strip()}")
            except Exception as e:
                st.error(f"❌ SLM (Qwen2) failed: {str(e)}")

        with st.spinner("Testing LLM (Llama 3.1)..."):
            try:
                llm_resp = test_client.chat.completions.create(
                    model=HF_MODELS[LLM_MODEL],
                    messages=[{"role": "user", "content": "say hi"}],
                    max_tokens=5
                )
                st.success(f"✅ LLM connected! → {llm_resp.choices[0].message.content.strip()}")
            except Exception as e:
                st.error(f"❌ LLM (Llama 3.1) failed: {str(e)}")

        st.divider()
        st.markdown("**Routing Logic**")
        st.markdown(
            "- Academic CS query → **SLM** (fast, KB-grounded)\n"
            "- Non-academic / Mixed → **LLM** (full reasoning)\n"
            "- KB search latency always < 0.01s"
        )

# Load KB
try:
    kb_df = load_kb()
    st.sidebar.success(f"✅ KB loaded ({len(kb_df)} entries)")
except FileNotFoundError:
    st.sidebar.error("❌ knowledge_dataset_llm.csv not found. Add it to your repo.")
    st.stop()

# Main Input
st.subheader("Ask a Question")
query = st.text_area(
    "Enter your query:",
    placeholder="e.g. What is a Binary Search Tree and what are its advantages?",
    height=100
)

if st.button("🚀 Submit", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    elif "hf_YOUR_TOKEN_HERE" in HF_TOKEN:
        st.error("⚠️ Please set your HuggingFace token in the sidebar.")
    else:
        with st.spinner("Routing and generating response..."):
            response, model_used, meta, latency = academic_router(query, kb_df)

        # Route badge
        if model_used == "SLM":
            st.success("✅ Routed to **SLM** — Qwen2-1.5B (KB-grounded academic query)")
        else:
            st.info("🔁 Routed to **LLM** — Llama-3.1-8B (Non-academic or mixed query)")

        # Metadata
        with st.expander("🔍 Extracted Metadata"):
            c1, c2 = st.columns(2)
            c1.metric("Academic?",   "Yes ✅" if meta.get("is_academic") else "No ❌")
            c2.metric("Pairs Found", len(meta.get("queries", [])))

            if meta.get("queries"):
                st.markdown("**Topic → Intent Mapping:**")
                for pair in meta.get("queries", []):
                    st.markdown(f"- **{pair['topic']}** → `{pair['intent']}`")
            else:
                st.write("No academic topic-intent pairs extracted.")

        # Latency
        with st.expander("⏱️ Latency Breakdown"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total",      f"{latency['total']:.2f}s")
            c2.metric("Extraction", f"{latency['extraction']:.2f}s")
            c3.metric("KB Search",  f"{latency['kb_search']:.4f}s")
            c4.metric("Formatting", f"{latency['formatting']:.2f}s")

            if model_used == "SLM":
                st.info(
                    "ℹ️ **Demo Note:** Running on HuggingFace free-tier API (~5–8s network overhead per call). "
                    "Local GPU benchmark: SLM **7–13s** vs LLM **60–90s** (6–8× improvement)."
                )
            else:
                st.info(
                    "ℹ️ **Demo Note:** Running on HuggingFace free-tier API. "
                    "Local GPU benchmark: LLM queries averaged **60–90s**."
                )

        # Response
        st.subheader("📄 Response")
        if response.startswith(("HF_", "CONNECTION_", "MODEL_")):
            st.error(response)
        else:
            st.markdown(response)

st.divider()
st.caption("Mindful Queries — SRM Institute of Science and Technology")
