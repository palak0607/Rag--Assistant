import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="PDF RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> RAGEngine
sessions: dict[str, dict] = {}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str
    processing_time: float


class SessionInfo(BaseModel):
    session_id: str
    filename: str
    num_chunks: int
    message: str


@app.get("/")
def root():
    return {"status": "running", "message": "PDF RAG Assistant API"}


@app.post("/upload", response_model=SessionInfo)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and create a RAG session."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    # Save file
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 20MB limit.")

    with open(file_path, "wb") as f:
        f.write(content)

    # Build RAG index
    try:
        engine = RAGEngine()
        num_chunks = engine.ingest_pdf(str(file_path))
        # engine.ingest_documents([str(file_path)])

        # num_chunks = len(engine.all_chunks)

        sessions[session_id] = {
            "engine": engine,
            "filename": file.filename,
            "num_chunks": num_chunks,
            "created_at": time.time(),
        }

    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return SessionInfo(
        session_id=session_id,
        filename=file.filename,
        num_chunks=num_chunks,
        message=f"PDF processed successfully into {num_chunks} chunks.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Ask a question about the uploaded PDF."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    session = sessions[request.session_id]
    engine: RAGEngine = session["engine"]

    start = time.time()
    try:
        result = engine.query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    processing_time = round(time.time() - start, 2)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
        processing_time=processing_time,
    )


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get session info."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = sessions[session_id]
    return {
        "session_id": session_id,
        "filename": s["filename"],
        "num_chunks": s["num_chunks"],
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"message": "Session deleted successfully."}


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions)}
