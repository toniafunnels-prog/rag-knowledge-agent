"""
FastAPI server for the Knowledge-Base Q&A Agent.

Endpoints:
  POST /ingest    -> (re)ingest all documents in /data
  POST /ask       -> ask a question, get a grounded answer + sources
  GET  /health    -> basic health check
  GET  /          -> minimal chat UI (see static/index.html)
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from .rag_pipeline import RAGPipeline

app = FastAPI(title="Knowledge-Base Q&A Agent")
pipeline = RAGPipeline()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    count = pipeline.ingest_directory()
    return {"chunks_ingested": count}


@app.post("/ask")
def ask(req: AskRequest):
    result = pipeline.answer(req.question)
    return result


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
