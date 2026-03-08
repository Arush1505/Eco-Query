# 🧠 Mindful Queries
### Energy-Aware Academic Query Routing using Small & Large Language Models

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-yellow.svg)](https://huggingface.co)
[![Research](https://img.shields.io/badge/Type-Research%20Project-green.svg)](#)

> A hybrid query routing system that intelligently directs academic queries to a Small Language Model (SLM) and complex/non-academic queries to a Large Language Model (LLM) — reducing energy consumption by up to **95%** and latency by **6–8×** for invariant academic workloads.

---

## 📌 Overview

Most AI-powered Q&A systems send every query to a large LLM regardless of complexity. For academic domains like Data Structures and Algorithms, the majority of questions are **static, repetitive, and knowledge-based** — they don't require full LLM reasoning.

**Mindful Queries** solves this by:
1. Classifying each query as academic or non-academic
2. Extracting topic-intent pairs from the query
3. Validating topics against a structured knowledge base
4. Routing validated academic queries → **SLM (Qwen2-1.5B)**
5. Routing complex/mixed/non-academic queries → **LLM (Llama-3.1-8B)**

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Query Normalization   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   SLM Metadata          │  ← Qwen2-1.5B extracts
│   Extraction            │    topic-intent pairs
└────────────┬────────────┘
             │
      is_academic?
       /         \
     No           Yes
      │             │
      ▼             ▼
   LLM Path    KB Validation
  (Llama 8B)  (Fuzzy Matching)
                   │
          fully grounded?
            /         \
          No           Yes
           │             │
           ▼             ▼
        LLM Path     SLM Path
       (Llama 8B)  (Qwen2-1.5B)
                   Response Synthesis
```

---

## ⚡ Key Results

| Metric | Baseline (LLM only) | Mindful Queries | Improvement |
|---|---|---|---|
| Latency (Academic) | 68.24s | 9.66s | **85.84% faster** |
| Power Draw | 18W (8B model) | 6W (1.5B model) | **66.67% less** |
| Energy per query | 1228.32 J | 57.96 J | **95.28% savings** |
| Routing Accuracy | — | 92.3% | — |

*Benchmarked on NVIDIA RTX 3050 4GB · Ollama inference engine*

---

## 🗂️ Project Structure

```
mindful-queries/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── knowledge_dataset_llm.csv # Structured academic knowledge base
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- HuggingFace account + API token ([get one here](https://huggingface.co/settings/tokens))
- Access to [Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on HuggingFace

### Installation

```bash
# Clone the repository
git clone https://github.com/your_username/mindful-queries.git
cd mindful-queries

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### Configuration

1. Open the app in your browser (usually `http://localhost:8501`)
2. Paste your HuggingFace token in the sidebar
3. Click **"Test HF Connection"** to verify both models
4. Start querying!

---

## 💡 Example Queries

**Academic queries (routed to SLM):**
```
What is a Binary Search Tree and what are its advantages?
Explain the Breadth-First Search algorithm with its typical applications.
What are the time and space complexities of Quick Sort?
What are advantages of Stack and Queue?
```

**Mixed queries (routed to LLM):**
```
Explain Dijkstra's algorithm and also tell me the best way to clean a cast iron skillet.
What are the advantages of a B-Tree, and how do I change a flat tire?
```

**Non-academic queries (routed to LLM):**
```
Who won the FIFA World Cup in 2022?
What is the best recipe for sourdough bread?
```

---

## 🧩 How It Works

### 1. Topic-Intent Pair Extraction
Instead of extracting flat lists of topics and intents, the system extracts **paired mappings**:
```json
{
  "queries": [
    {"topic": "Binary Tree",  "intent": "advantages"},
    {"topic": "Arrays",       "intent": "definition"}
  ],
  "is_academic": true
}
```
This ensures each topic is answered with its **specific** requested intent only.

### 2. Multi-Pass KB Validation
Topics go through three passes:
- **Pass 1:** Strong fuzzy match on heading/alias (threshold: 70)
- **Pass 2:** Attribute token matching against validated rows
- **Pass 3:** Lower-threshold extras-wide search (threshold: 60)

### 3. Mixed Query Detection
If any extracted topic cannot be grounded in the knowledge base, the entire query is routed to the LLM — preserving response quality.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| SLM | Qwen2-1.5B-Instruct (via HuggingFace) |
| LLM | Llama-3.1-8B-Instruct (via HuggingFace) |
| Fuzzy Matching | thefuzz (Levenshtein distance) |
| Text Normalization | NLTK WordNetLemmatizer |
| Knowledge Base | Structured CSV |
| API Client | OpenAI-compatible HuggingFace Router |

---

## 📊 Knowledge Base

The current KB covers core **Data Structures & Algorithms** concepts including:

- Linear structures: Arrays, Linked Lists, Stacks, Queues
- Trees: Binary Tree, BST, AVL Tree, Red-Black Tree, B-Tree
- Graphs: BFS, DFS, Dijkstra's Algorithm
- Sorting: Quick Sort, Merge Sort, Heap Sort
- Hashing: Hash Maps, Collision handling

Each entry contains: `definition · advantages · disadvantages · applications · code · extras`

> ⚠️ The dataset is actively being expanded to cover more CS domains.

---

## 📄 Research Paper

This project is based on the research paper:

**"Mindful Queries: Leveraging Small Language Models for Invariant Data"**
*Arush Jauhari, Mridangam Goswami, Dr. Prakash U M*
*SRM Institute of Science and Technology, Chennai*

---

## 🙋 Author

**Arush Jauhari**
- 🔗 [LinkedIn](https://www.linkedin.com/in/arush-jauhari-b350372a2/)
- 🎓 SRM Institute of Science and Technology, Chennai

---

## ⚠️ Limitations

- Knowledge base currently limited to Data Structures & Algorithms
- Deployed demo uses HuggingFace free-tier API (adds ~5–8s network overhead)
- SLM may occasionally misclassify edge-case mixed queries
- Benchmark results were measured on local GPU — HF API latency differs

---

## 📜 License

This project is for academic and research purposes.
