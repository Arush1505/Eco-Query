# 🧠 Eco-Query
### Energy-Aware Academic Query Routing using Small & Large Language Models

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference%20API-yellow.svg)](https://huggingface.co)
[![Research](https://img.shields.io/badge/Type-Research%20Project-green.svg)](#)

> A hybrid query routing system that intelligently directs academic queries to a Small Language Model (SLM) and complex/non-academic queries to a Large Language Model (LLM) — reducing energy consumption by up to **95%** and latency by **6–8×** for invariant academic workloads.

---

## 📌 Overview

Most AI-powered Q&A systems send every query to a large LLM regardless of complexity. For academic domains like Data Structures and Algorithms, the majority of questions are **static, repetitive, and knowledge-based** — they don't require full LLM reasoning.

**Eco-Query** solves this by:
1. Classifying each query as academic or non-academic
2. Extracting topic-intent pairs from the query
3. Validating topics against a structured knowledge base
4. Routing validated academic queries → **SLM (Qwen2-1.5B)**
5. Routing complex/mixed/non-academic queries → **LLM (Llama-3.1-8B)**


**Note : All these metrics were tested on LOCAL GPU , on actual production servers they might vary. The cost of renting VM is high , so for demo purpose I used huggingFace API**
---

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


## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| SLM | Qwen2-1.5B-Instruct (via HuggingFace) |
| LLM | Llama-3.1-8B-Instruct (via HuggingFace) |
| Knowledge Base | Structured CSV |
| API Client | OpenAI-compatible HuggingFace Router |

---


---

## 📄 Research Paper

This project is research based, not supposed to be production ready.
---

## ⚠️ Limitations

- Knowledge base currently limited to Data Structures & Algorithms
- Deployed demo uses HuggingFace free-tier API (adds ~5–8s network overhead)
- SLM may occasionally misclassify edge-case mixed queries
- Benchmark results were measured on local GPU — HF API latency differs

---
