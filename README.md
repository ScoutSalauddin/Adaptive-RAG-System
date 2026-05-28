# 🤖 Adaptive RAG System

> A production-grade, agentic question-answering system that thinks before it retrieves.

[![Python](https://img.shields.io/badge/python-3.9%2B-informational)](https://python.org)
[![Built with LangGraph](https://img.shields.io/badge/built%20with-LangGraph-orange)](https://python.langchain.com/langgraph/)
[![Vector DB](https://img.shields.io/badge/vector%20db-Qdrant-blueviolet)](https://qdrant.tech)
[![API](https://img.shields.io/badge/api-FastAPI-009688)](https://fastapi.tiangolo.com)
[![UI](https://img.shields.io/badge/ui-Streamlit-ff4b4b)](https://streamlit.io)

---

## What is this?

Most RAG systems are dumb pipelines — every query hits the vector store, even when the document is irrelevant or doesn't exist. This system is different.

Before doing any retrieval, it **classifies** what kind of question you're asking. Then it routes to the right strategy:

| Query Type | What happens |
|---|---|
| `index` | Searches your uploaded documents via Qdrant |
| `general` | Answers directly using GPT-4o's world knowledge |
| `search` | Runs a live Tavily web search for real-time info |

If retrieved documents don't pass a relevance check, the query gets **rewritten** and retried automatically. The whole flow is orchestrated as a stateful graph using LangGraph.

---

## How it works — the full pipeline

```
User Input
    │
    ▼
┌─────────────────────┐
│   Query Classifier  │  ← decides: index / general / search
└──────────┬──────────┘
           │
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
 Retriever    General LLM    Web Search
 (Qdrant)    (GPT-4o)        (Tavily)
     │
     ▼
 Relevance Grader
     │
  passed?
  ├── YES → Generator → Response
  └── NO  → Query Rewriter → Retriever (retry)
```

Every node in this graph is independently testable and swappable. The graph state is carried through each step, so context is never lost mid-pipeline.

---

## Codebase layout

```
.
├── src/
│   ├── main.py                    # FastAPI app bootstrap
│   ├── api/routes.py              # /rag/query and /rag/documents/upload
│   ├── config/
│   │   ├── settings.py            # Env-based config (pydantic)
│   │   └── prompts.yaml           # All LLM prompts in one place
│   ├── rag/
│   │   ├── graph_builder.py       # Assembles the LangGraph pipeline
│   │   ├── nodes.py               # One function per graph node
│   │   ├── retriever_setup.py     # Qdrant collection + embeddings
│   │   ├── document_upload.py     # Chunking, embedding, indexing
│   │   └── reAct_agent.py         # ReAct agent for tool use
│   ├── models/                    # Pydantic schemas (state, requests, grades)
│   ├── memory/                    # MongoDB + in-memory chat history
│   ├── llms/openai.py             # GPT-4o client init
│   ├── db/mongo_client.py         # Async MongoDB via Motor
│   ├── core/                      # Logger, base config
│   └── tools/                     # Graph routing helpers, shared utils
│
└── streamlit_app/
    ├── home.py                    # Login / signup page
    ├── pages/chat.py              # Main chat UI + document upload sidebar
    └── utils/api_client.py        # HTTP client wrapping the FastAPI backend
```

---

## Getting started

### Prerequisites

- Python 3.9 or higher
- A running [Qdrant](https://qdrant.tech/documentation/quick-start/) instance
- MongoDB (local or Atlas)
- API keys: OpenAI, Tavily

### Installation

```bash
git clone https://github.com/ScoutSalauddin/Adaptive-RAG-System.git
cd Adaptive-RAG-System

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment setup

Copy this into a `.env` file at the project root and fill in your keys:

```env
OPENAI_API_KEY=
TAVILY_API_KEY=

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_DOCS_COLLECTION=documents
QDRANT_CODE_COLLECTION=code_documents

MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=adaptive_rag
```

> Never commit `.env` to version control. It's already in `.gitignore`.

### Running locally

Open two terminals:

```bash
# Terminal A — backend
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Terminal B — frontend
streamlit run streamlit_app/home.py
```

Then open:
- **Chat UI** → http://localhost:8501
- **Swagger docs** → http://localhost:8000/docs

---

## API reference

### `POST /rag/query`

Send a question and get a response routed through the adaptive pipeline.

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the uploaded report", "session_id": "abc123"}'
```

```json
{
  "result": {
    "type": "ai",
    "content": "The report covers..."
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `query` | `str` | The user's question |
| `session_id` | `str` | Used to load/save conversation history |

---

### `POST /rag/documents/upload`

Index a PDF or TXT file into the vector store.

```bash
curl -X POST http://localhost:8000/rag/documents/upload \
  -H "X-Description: Q3 financial report" \
  -F "file=@report.pdf"
```

```json
{ "status": true }
```

| Input | Notes |
|---|---|
| `X-Description` header | Short description stored with the document |
| `file` (form field) | `.pdf` or `.txt` only |

---

## Graph node reference

Each node is a standalone Python function in `src/rag/nodes.py`:

| Node | Input | What it does |
|---|---|---|
| `query_analysis` | Raw query | Classifies as `index`, `general`, or `search` |
| `retriever` | Query + collection | Fetches top-k chunks from Qdrant |
| `grade` | Query + chunks | Scores each chunk for relevance; filters out noise |
| `rewrite` | Query | Reformulates the query if grading failed |
| `generate` | Query + context | Calls GPT-4o to produce the final answer |
| `web_search` | Query | Runs Tavily search; returns structured results |
| `general_llm` | Query | Direct GPT-4o completion, no retrieval |

---

## Configuration reference

All prompts live in `src/config/prompts.yaml` — edit them without touching Python:

```yaml
classify_prompt:   # Instructs the model to output index/general/search
grading_prompt:    # Binary relevance judgment per retrieved chunk
rewrite_prompt:    # Improves a query that failed grading
generate_prompt:   # Final answer synthesis from context
system_prompt:     # ReAct agent system instructions
```

All environment-driven settings are in `src/config/settings.py` as a Pydantic `BaseSettings` model — strongly typed, validated on startup.

---

## Deployment

**Single-process (dev):**
```bash
uvicorn src.main:app --reload
```

**Multi-worker (production):**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

For containerised deployment, add a `Dockerfile` and a `docker-compose.yml` that spins up this app alongside Qdrant and MongoDB.

---

## Design decisions worth noting

**Why LangGraph over a simple chain?**  
Chains are linear. This pipeline needs conditional branching (grade → rewrite → retry), which requires a stateful graph. LangGraph makes that explicit and debuggable.

**Why Qdrant over FAISS?**  
Qdrant is a proper server with a REST API, persistence, and payload filtering. FAISS is an in-process library — great for prototypes, harder to scale or share across services.

**Why MongoDB for chat history?**  
Conversations need to survive server restarts and scale across workers. An in-memory store won't survive a dyno restart. MongoDB with Motor gives us async-native persistence without overhead.

---

## Roadmap

- [ ] Docker Compose (app + Qdrant + MongoDB in one command)
- [ ] Support for additional LLM providers (Anthropic, Gemini)
- [ ] Streaming responses via SSE
- [ ] Per-user document namespacing
- [ ] Evaluation harness with retrieval quality metrics
- [ ] Usage analytics dashboard

---

## Contributing

PRs are welcome. Please:

1. Fork and branch off `main`
2. Follow `CODE_STYLE_GUIDE.md` — PEP 8, docstrings, type hints
3. Write or update tests for changed behaviour
4. Open a PR with a clear description of what changed and why

---

## Supporting docs

| Document | Purpose |
|---|---|
| [`CODE_STYLE_GUIDE.md`](CODE_STYLE_GUIDE.md) | Formatting and naming conventions |
| [`QDRANT_SETUP_GUIDE.md`](QDRANT_SETUP_GUIDE.md) | Step-by-step Qdrant setup |
| [`DOCUMENT_UPLOAD_FLOW.md`](DOCUMENT_UPLOAD_FLOW.md) | How documents get chunked and indexed |
| [`DOCUMENT_FLOW_VISUAL.md`](DOCUMENT_FLOW_VISUAL.md) | Visual walkthrough of the ingestion pipeline |
| [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) | Common patterns, snippets, and tips |
| [`DOCUMENTATION_INDEX.md`](DOCUMENTATION_INDEX.md) | Full docs navigation |

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph ~0.5.4 |
| LLM | OpenAI GPT-4o (via LangChain ~0.3.27) |
| Vector store | Qdrant |
| Web search | Tavily |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Chat persistence | MongoDB (Motor async driver) |
| Validation | Pydantic ~2.11.7 |

---

## Author

Built and maintained by [@ScoutSalauddin](https://github.com/ScoutSalauddin).

---

## License

MIT — see [`LICENSE`](LICENSE).
