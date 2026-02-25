# DocMind — PDF RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you upload any PDF and have an intelligent conversation with it.

## Architecture

```
PDF Upload
    │
    ▼
PyPDF Loader ──► Text Splitter (RecursiveCharacterTextSplitter)
                      │  chunk_size=1000, overlap=200
                      ▼
              OpenAI Embeddings (text-embedding-3-small)
                      │
                      ▼
                 FAISS Vector Store
                      │
              ┌───────┴────────┐
          User Query       MMR Search (top-k=4)
              │                │
              ▼                ▼
        Embed Query ──► Retrieved Chunks
                              │
                              ▼
                    GPT-4o-mini + Memory
                    (ConversationBufferWindowMemory k=5)
                              │
                              ▼
                    Answer + Source Attribution
```

## Advanced Features

| Feature | Implementation |
|---|---|
| **MMR Search** | Maximal Marginal Relevance reduces redundant chunks |
| **Conversational Memory** | Remembers last 5 Q&A pairs for follow-up questions |
| **Source Attribution** | Every answer cites exact page numbers |
| **Session Management** | Multi-user support with isolated sessions |
| **Recursive Splitting** | Sentence-aware splitting preserves context |
| **Efficient Embeddings** | `text-embedding-3-small` — cost-efficient, high quality |

## Tech Stack

- **Backend:** FastAPI + LangChain + FAISS + OpenAI
- **Frontend:** React 18 + react-dropzone + react-markdown
- **Deployment:** Docker + Docker Compose

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and configure
git clone <repo>
cd rag-assistant
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENAI_API_KEY

# Run
docker-compose up --build
```

Visit `http://localhost:3000`

### Option 2: Manual

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF, returns session_id |
| `POST` | `/chat` | Ask a question |
| `GET` | `/session/{id}` | Get session info |
| `DELETE` | `/session/{id}` | Delete session |
| `GET` | `/health` | Health check |

## Design Decisions

**Why FAISS over Pinecone/Weaviate?**
For a demo/single-server setup, FAISS runs entirely in-memory with no external dependencies. At enterprise scale, you'd swap to a managed vector DB with persistent storage.

**Why chunk_size=1000, overlap=200?**
After experimentation, 1000 tokens captures enough context per chunk without overwhelming the retriever. 200-token overlap ensures no information is lost at chunk boundaries.

**Why MMR search?**
Standard similarity search often returns nearly-identical chunks. MMR balances relevance with diversity, giving the LLM a richer, less redundant context window.

**Why GPT-4o-mini?**
Optimal cost/quality ratio for Q&A tasks. For complex reasoning or very technical documents, swap to `gpt-4o`.

## Environment Variables

```env
OPENAI_API_KEY=sk-...
```

## Production Considerations

- Replace in-memory session store with Redis
- Replace local FAISS with Pinecone/Weaviate for persistence
- Add authentication (JWT/OAuth)
- Add rate limiting
- Store uploaded files in S3
- Add async processing for large PDFs



QUESTIONS:
Summarize the key highlights of this report
Compare Q1 2025 vs Q2 2025 total revenue
What was Tesla's GAAP operating income in Q2 2025?

"What was Tesla's total revenue in Q2 2025?"
"What is Tesla's Robotaxi strategy?"
"How did operating income change from Q2 2024 to Q2 2025?"
"What is Tesla's outlook for new vehicles in 2025?"




