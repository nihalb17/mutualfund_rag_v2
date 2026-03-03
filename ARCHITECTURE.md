# Mutual Fund FAQ RAG Chatbot — Architecture

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers factual questions about three specific Axis Mutual Fund schemes using Gemini LLM. The chatbot is grounded exclusively in data scraped from three Groww pages, returns citation links with every answer, and enforces strict guardrails against out-of-scope questions and investment advice.

### Knowledge Base (Source URLs)

| Scheme | Source URL |
|---|---|
| Axis Liquid Direct Fund Growth | https://groww.in/mutual-funds/axis-liquid-direct-fund-growth |
| Axis ELSS Tax Saver Direct Plan Growth | https://groww.in/mutual-funds/axis-elss-tax-saver-direct-plan-growth |
| Axis Flexi Cap Fund Direct Growth | https://groww.in/mutual-funds/axis-flexi-cap-fund-direct-growth |

### Key LLM

**Google Gemini** (via `google-generativeai` Python SDK)

---

## Data Fields Available Per Scheme

Each Groww page exposes the following fields that the chatbot can answer questions about:

- **NAV** (Net Asset Value)
- **Fund Size** (AUM)
- **Expense Ratio**
- **Minimum SIP Amount**
- **Minimum Lump Sum Investment**
- **Exit Load** (conditions and percentage)
- **Stamp Duty** (percentage)
- **Tax Implication** (STCG / LTCG)
- **Risk Level** (Riskometer)
- **Fund Category & Sub-category**
- **Fund Manager(s)** (name, tenure)
- **Advanced Ratios** (Sharpe Ratio, Sortino Ratio, Standard Deviation, Beta, Alpha)
- **Returns** (1Y, 3Y, 5Y, annualised / absolute)
- **Investment Objective**
- **Fund House**
- **Launch Date / Inception Date**
- **Lock-in Period** (ELSS specific)

---

## System Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                   User (Browser/UI)                  │
└─────────────────────────┬────────────────────────────┘
                          │ HTTP (REST)
┌─────────────────────────▼────────────────────────────┐
│              FastAPI Backend (Python)                 │
│  ┌───────────────────────────────────────────────┐   │
│  │              Chat Session Manager             │   │
│  │  (maintains per-session conversation history) │   │
│  └───────────────────────┬───────────────────────┘   │
│  ┌───────────────────────▼───────────────────────┐   │
│  │                 RAG Pipeline                  │   │
│  │  1. Guardrail Check (scope / advice filter)   │   │
│  │  2. Query Embedding (Gemini Embeddings)       │   │
│  │  3. Vector Store Retrieval (ChromaDB)         │   │
│  │  4. Context Assembly + Citation Tracking      │   │
│  │  5. Prompt Construction (system + history)    │   │
│  │  6. Gemini LLM Generation                     │   │
│  │  7. Response + Citations returned to UI       │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
          │                             │
┌─────────▼──────────┐       ┌──────────▼────────────┐
│   ChromaDB Vector  │       │   Gemini LLM API       │
│   Store (local)    │       │  (google-generativeai) │
│   + Metadata Store │       └───────────────────────┘
│     (source URLs)  │
└────────────────────┘
```

---

## Phase-wise Breakdown

---

## Phase 1 — Data Ingestion & Vector Store Setup

### Goal
Scrape the three Groww mutual fund pages, parse structured data, chunk it, embed it using Gemini Embeddings, and store it in a local ChromaDB vector store. Each chunk must carry the source URL as metadata for citation generation.

### Scope
- Web scraping (static HTML parsing with `requests` + `BeautifulSoup`)
- Text cleaning and structured chunking strategy
- Embedding via `models/text-embedding-004` (Gemini)
- ChromaDB as local persistent vector store
- Metadata tagging: `source_url`, `scheme_name`, `field_category`

### Directory Structure
```
mutualfund_rag_v2/
├── phase_1/
│   ├── scraper.py          # Scrapes each Groww URL
│   ├── parser.py           # Parses raw HTML into structured fields
│   ├── chunker.py          # Splits data into chunks with metadata
│   ├── embedder.py         # Embeds chunks using Gemini Embeddings
│   ├── vector_store.py     # Initialises and populates ChromaDB
│   └── run_ingestion.py    # Orchestrator script for Phase 1
├── data/
│   ├── raw/                # Raw scraped HTML files (per scheme)
│   └── chunks/             # Serialised chunks as JSON (for debugging)
├── vector_db/              # ChromaDB persistent storage directory
├── .env                    # GEMINI_API_KEY
└── requirements.txt
```

### Chunking Strategy
Each chunk is a logical unit tied to a specific *field* of a *scheme*:

| Chunk Type | Example Content |
|---|---|
| Fund metadata | "Axis Liquid Direct Fund: NAV ₹2,850.12, AUM ₹xxxx Cr, Expense Ratio 0.15%" |
| Exit load + Tax | "Exit load: Nil. Stamp Duty: 0.005%. STCG taxed at slab rate." |
| Fund manager | "Fund Manager: Devang Shah, Jan 2013 – Present" |
| Advanced ratios | "Sharpe Ratio: x.xx, Beta: x.xx, Sortino: x.xx" |
| Returns | "1Y Return: x.xx%, 3Y Return: x.xx%, 5Y Return: x.xx%" |
| About / Objective | "Investment Objective: To generate income consistent with..." |

Each chunk also carries:
```json
{
  "scheme_name": "Axis Liquid Direct Fund Growth",
  "source_url": "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth",
  "field_category": "expense_ratio"
}
```

### Deliverables
- Populated ChromaDB collection (`mutualfund_kb`)
- JSON dump of all chunks in `data/chunks/` for verification
- `requirements.txt` with all dependencies

### Test Cases — Phase 1

| # | Test | How to Run | Pass Condition |
|---|---|---|---|
| T1.1 | Scraper fetches all 3 URLs successfully | `python phase_1/run_ingestion.py` | No HTTP errors; raw HTML saved in `data/raw/` |
| T1.2 | NAV, Expense Ratio, Fund Size parsed correctly for all 3 schemes | `python phase_1/parser.py --debug` | Parsed values match values visible on Groww pages |
| T1.3 | Each chunk has `source_url`, `scheme_name`, `field_category` in metadata | `python phase_1/chunker.py --validate` | All chunks have 3 required metadata keys |
| T1.4 | Embeddings generated for all chunks | `python phase_1/embedder.py --test` | Non-zero float vectors returned for every chunk |
| T1.5 | ChromaDB collection has correct document count | `python phase_1/vector_store.py --count` | Count equals total number of chunks |
| T1.6 | Similarity search returns relevant chunk | Manual Python shell test: query "expense ratio liquid fund" | Top result chunk belongs to Axis Liquid Direct Fund |
| T1.7 | Metadata filtering works (filter by scheme) | Query DB with `where={"scheme_name": "Axis Liquid Direct Fund Growth"}` | Only chunks from that scheme are returned |

---

## Phase 2 — RAG Pipeline Core

### Goal
Build the core RAG pipeline that takes a user query, retrieves relevant chunks from ChromaDB, assembles a context-aware prompt with strict guardrails, calls Gemini LLM, and returns a factual answer with citation links.

### Scope
- Query embedding + vector retrieval
- Guardrail logic (3 modes: out-of-scope, investment advice, unknown scheme)
- Prompt engineering (system prompt with strict fact-only instructions)
- Gemini LLM call (`gemini-1.5-flash` or `gemini-1.5-pro`)
- Response formatting: answer text + list of citation URLs

### Directory Structure
```
mutualfund_rag_v2/
├── phase_2/
│   ├── retriever.py        # ChromaDB similarity search + metadata filtering
│   ├── guardrails.py       # Input validation: scope, scheme, advice detection
│   ├── prompt_builder.py   # Assembles system prompt + context + query
│   ├── llm_client.py       # Gemini API wrapper
│   ├── rag_pipeline.py     # End-to-end RAG orchestrator
│   └── test_rag.py         # Phase 2 test suite
```

### Guardrail Logic

```
User Query
    │
    ▼
[1] Is it investment advice? ──YES──► "I am only here to provide information regarding mutual funds."
    │ NO
    ▼
[2] Does it mention a scheme outside the 3 known schemes? ──YES──► "I don't have information regarding the scheme."
    │ NO
    ▼
[3] Retrieve top-k chunks from ChromaDB
    │
    ▼
[4] Does retrieved context contain relevant info? ──NO──► "I don't have an answer to the question you are asking."
    │ YES
    ▼
[5] Build prompt → Call Gemini → Return answer + citations
```

### Known Schemes Registry (for guardrail #2)
```python
KNOWN_SCHEMES = {
    "axis liquid direct fund": "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth",
    "axis elss tax saver": "https://groww.in/mutual-funds/axis-elss-tax-saver-direct-plan-growth",
    "axis flexi cap fund": "https://groww.in/mutual-funds/axis-flexi-cap-fund-direct-growth",
}
```

### System Prompt Template

```
You are a factual Mutual Fund information assistant. Your only job is to answer
questions about the following three Axis Mutual Fund schemes based strictly on
the provided context. 

Rules:
1. ONLY answer from the provided context. Do not use any outside knowledge.
2. Do not give investment advice, buy/sell recommendations, or opinions.
3. If the context does not contain the answer, say: "I don't have an answer to 
   the question you are asking."
4. Always cite only the source URLs from which the answer was derived.
5. Present only facts — numbers, names, dates, percentages — as stated in context.

Context:
{retrieved_context}

Conversation History:
{conversation_history}

User Question: {user_query}

Answer (facts only, cite sources):
```

### Citation Mechanism
- Each retrieved chunk carries its `source_url` in metadata.
- After LLM responds, the pipeline collects the unique `source_url` values of all chunks used in context construction and returns them as a `citations` list alongside the answer.
- The UI renders these as clickable links.

### Deliverables
- `rag_pipeline.py` with `query(question: str, session_history: list) → dict` function
- Response schema:
  ```json
  {
    "answer": "Expense ratio of Axis Liquid Direct Fund is 0.15%.",
    "citations": [
      "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth"
    ],
    "guardrail_triggered": false
  }
  ```

### Test Cases — Phase 2

| # | Test | Input | Expected Output |
|---|---|---|---|
| T2.1 | NAV factual query | "What is the NAV of Axis Liquid Fund?" | Answer contains NAV value + citation URL |
| T2.2 | Expense ratio query | "What is the expense ratio of Axis ELSS Tax Saver?" | Correct expense ratio + citation URL |
| T2.3 | Exit load query | "What is the exit load of Axis Flexi Cap Fund?" | Correct exit load info + citation URL |
| T2.4 | Fund manager query | "Who manages the Axis Liquid Fund?" | Fund manager name(s) + tenure + citation URL |
| T2.5 | Minimum SIP query | "What is the minimum SIP for Axis ELSS?" | Correct SIP amount + citation URL |
| T2.6 | Investment advice guardrail | "Should I invest in Axis Liquid Fund?" | Guardrail message about providing information only |
| T2.7 | Out-of-scope scheme | "Tell me about Mirae Asset Liquid Fund" | "I don't have information regarding the scheme." |
| T2.8 | Completely out-of-scope question | "What is the capital of France?" | "I don't have an answer to the question you are asking." |
| T2.9 | Tax implication query | "What are the tax implications for Axis Flexi Cap Fund?" | STCG/LTCG facts + stamp duty + citation URL |
| T2.10 | Multi-scheme query with citations | "Compare expense ratios of all three funds" | Data from all 3 + 3 citation URLs |
| T2.11 | Citation accuracy | Any factual query | Only relevant source URLs included, not all 3 blindly |

---

## Phase 3 — (REMOVED) Conversation Context Management

> [!NOTE]
> Phase 3 was removed to improve system performance. The chatbot now operates in a stateless manner.

---

---

## Phase 4 — FastAPI Backend

### Goal
Wrap the RAG pipeline and session manager into a production-ready REST API using FastAPI with CORS support for the frontend.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/session/new` | Create a new chat session |
| `DELETE` | `/session/{session_id}` | Clear a session's history |
| `GET` | `/health` | Health check |

### Request / Response Schema

**POST /chat**
```json
// Request
{
  "session_id": "abc123",
  "message": "What is the NAV of Axis Liquid Fund?"
}

// Response
{
  "session_id": "abc123",
  "answer": "The NAV of Axis Liquid Direct Fund Growth is ₹2,850.12 as of...",
  "citations": [
    "https://groww.in/mutual-funds/axis-liquid-direct-fund-growth"
  ],
  "guardrail_triggered": false
}
```

### Directory Structure
```
mutualfund_rag_v2/
├── phase_4/
│   ├── main.py             # FastAPI app with all route definitions
│   ├── schemas.py          # Pydantic request/response models
│   ├── dependencies.py     # Shared app-level dependencies (DB connection, etc.)
│   └── test_api.py         # Phase 4 API test suite (using httpx)
```

### Test Cases — Phase 4

| # | Test | How to Run | Pass Condition |
|---|---|---|---|
| T4.1 | Health check endpoint | `GET /health` | `{"status": "ok"}` returned |
| T4.2 | New session creation | `POST /session/new` | Returns `{"session_id": "<uuid>"}` |
| T4.3 | Valid chat request | `POST /chat` with valid session + message | Returns answer + citations JSON |
| T4.4 | Investment advice guardrail via API | `POST /chat` with "Should I invest in Axis Liquid Fund?" | Guardrail message returned, `guardrail_triggered: true` |
| T4.5 | Unknown scheme via API | `POST /chat` with unknown fund name | Unknown scheme message returned |
| T4.6 | Out-of-scope question via API | `POST /chat` with unrelated question | Out-of-scope message returned |
| T4.7 | Invalid session ID | `POST /chat` with non-existent `session_id` | 404 error or auto-create new session |
| T4.8 | Session deletion | `DELETE /session/{id}` then `POST /chat` for follow-up | No history available for follow-up |
| T4.9 | Concurrent sessions | Send simultaneous requests from 2 different sessions | Each session returns correct, isolated answers |
| T4.10 | CORS headers | Browser OPTIONS preflight request | Correct CORS headers present in response |

---

## Phase 5 — Frontend Chatbot UI

### Goal
Build a modern, responsive chatbot UI using plain **HTML, CSS, and JavaScript** that connects to the FastAPI backend via `fetch()` API calls, renders answers with clickable citation links, and maintains a visual chat history in the browser.

### Features
- Chat message bubbles (user vs assistant)
- Clickable citation links rendered below each assistant response
- "New Chat" button to reset session
- Loading spinner while awaiting response
- Guardrail messages styled distinctly (e.g., amber/warning color)
- Mobile-responsive layout

### Technology Choice
**Plain HTML + CSS + JavaScript** — No framework dependency; communicates with FastAPI backend via `fetch()` REST calls. Session ID stored in `sessionStorage`.

### Directory Structure
```
mutualfund_rag_v2/
├── phase_5/
│   ├── index.html          # Main chat UI page
│   ├── style.css           # Styling (chat bubbles, citations, guardrail states)
│   ├── app.js              # JS logic: fetch(), session management, DOM rendering
│   └── test_ui.py          # End-to-end UI smoke tests (Playwright)
```

### UI Component Layout
```
┌────────────────────────────────────────────┐
│  🏦 Mutual Fund FAQ Chatbot                 │
│  Powered by Gemini AI    [New Chat]         │
├────────────────────────────────────────────┤
│                                            │
│  [User]: What is the NAV of Axis ELSS?     │
│                                            │
│  [Bot]: The NAV of Axis ELSS Tax Saver     │
│  Direct Plan Growth is ₹xx.xx as of...     │
│                                            │
│  📎 Sources:                               │
│  • groww.in/mutual-funds/axis-elss-...     │
│                                            │
│  [User]: What about its expense ratio?     │
│                                            │
│  [Bot]: The expense ratio is 0.xx%...      │
│  📎 Sources: ...                           │
│                                            │
├────────────────────────────────────────────┤
│  [ Type your question here...      ] [Send]│
└────────────────────────────────────────────┘
```

### Test Cases — Phase 5

| # | Test | Steps | Pass Condition |
|---|---|---|---|
| T5.1 | App loads without errors | Open `index.html` in browser (with backend running) | UI renders correctly, no JS console errors |
| T5.2 | Factual question answered with citation | Type "What is the expense ratio of Axis Liquid Fund?" → Send | Answer displayed with clickable citation link |
| T5.3 | Citation link is correct | Click citation link | Browser navigates to correct Groww page |
| T5.4 | Guardrail message styled differently | Type "Should I buy Axis ELSS?" | Guardrail response appears in amber/warning style |
| T5.5 | Follow-up query resolved | Ask a fund question, then ask "What about its exit load?" | Second answer resolves to same fund correctly |
| T5.6 | New Chat clears history | Click "New Chat", then ask follow-up | No prior context; new session started |
| T5.7 | Loading state shown | Submit any question | Spinner/loader visible until response arrives |
| T5.8 | Multi-citation query | Ask cross-scheme question | Multiple citations rendered as separate links |
| T5.9 | Long conversation scroll | Ask 10+ questions | Chat window scrolls; older messages remain visible |

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`gemini-1.5-flash` / `gemini-1.5-pro`) |
| Embeddings | `models/text-embedding-004` (Gemini) |
| Vector Store | ChromaDB (local persistent) |
| Web Scraping | `requests` + `BeautifulSoup4` |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript (`fetch()`) |
| Session Storage | In-memory Python dict (per process) |
| Config | `python-dotenv` → `.env` file |
| Testing | `pytest` + `httpx` (API tests) |

---

## Environment Variables

```
# .env
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=./vector_db
MAX_HISTORY_TURNS=5
TOP_K_RETRIEVAL=5
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
```

---

## Dependency List (`requirements.txt`)

```
google-generativeai>=0.5.0
chromadb>=0.5.0
fastapi>=0.110.0
uvicorn>=0.29.0
requests>=2.31.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
httpx>=0.27.0
pytest>=8.0.0
pydantic>=2.0.0
```

---

## Development Sequence

```
Phase 1: Ingestion & Vector DB   ──► Working vector store with all 3 schemes
    │
    ▼
Phase 2: RAG Core Pipeline       ──► Query → Retrieve → Gemini → Answer + Citations
    │
    ▼
Phase 3: Conversation Context    ──► Follow-up query support with session history
    │
    ▼
Phase 4: FastAPI Backend         ──► REST API wrapping the RAG + Session pipeline
    │
    ▼
Phase 5: Chatbot UI              ──► HTML/CSS/JS UI connected to FastAPI backend via fetch()
```

---

## Guardrail Summary

| Scenario | Trigger Condition | Response to User |
|---|---|---|
| Investment advice | Query contains intent to invest, buy, sell, recommend, compare for investment purpose | "I am only here to provide information regarding mutual funds." |
| Unknown scheme | Query references a mutual fund not in the 3 known schemes | "I don't have information regarding the scheme." |
| Out-of-scope topic | Retrieval returns no relevant chunks; topic is entirely unrelated | "I don't have an answer to the question you are asking." |
| Factual answer only | Relevant chunks found | Answer + citations (only sources used) |
