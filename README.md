# Knowledge-Base Q&A Agent (RAG)

A retrieval-augmented generation agent that answers questions grounded in a
company's own documents — policies, wikis, SOPs, onboarding docs — with
citations back to source files. No hallucinated answers about internal
processes; if it's not in the docs, the agent says so.

**Portfolio niche:** Internal knowledge / document agents (part of the AI
Automation, Support & Data Agent Developer cluster.)

---

## Why this matters to a client (case study framing)

> **Problem:** Employees waste time searching scattered docs (Notion, Google
> Drive, PDFs) for policy answers, or interrupt HR/ops staff with repeat
> questions.
> **Solution:** A chat interface that answers instantly from the company's
> actual documents, with source citations for trust and auditability.
> **Measurable outcome (frame with real client data when deployed):**
> reduction in repeat HR/support tickets, faster new-hire ramp-up, fewer
> "where do I find X" Slack messages.

**Stack note:** This project is 100% Claude-ecosystem — Anthropic Claude
for generation, Voyage AI (Anthropic's recommended embedding partner) for
embeddings. No OpenAI dependency anywhere in the stack.

---

## Architecture

```
 ┌─────────────┐      ┌──────────────┐      ┌────────────────┐
 │  Documents   │ ---> │   Chunking + │ ---> │  Chroma Vector  │
 │ (.txt/.md)   │      │  Embedding   │      │  Store (local)  │
 └─────────────┘      │ (Voyage AI)  │      └────────────────┘
                       └──────────────┘
                                                     │
 User question ──> embed query (Voyage) ──> retrieve top-k chunks ──┘
                                          │
                                          v
                              ┌────────────────────────┐
                              │  Claude (Anthropic API)  │
                              │  answers using ONLY the  │
                              │  retrieved context        │
                              └────────────────────────┘
                                          │
                                          v
                              Grounded answer + source citations
```

- **Chunking:** simple character-based chunking with overlap (swap for
  recursive/semantic chunking for production-grade accuracy on longer docs).
- **Embeddings:** Voyage AI (`voyage-3.5-lite`) — Anthropic's recommended
  embedding provider, since Anthropic doesn't offer its own embedding model.
  Requires a free Voyage AI API key (voyageai.com).
- **Vector store:** Chroma, local/persistent — swap for Pinecone/Weaviate/
  pgvector for multi-user or larger-scale deployments.
- **Generation:** Claude via Anthropic API, with a strict grounding system
  prompt that forces citation and refuses to answer outside the provided
  context.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_anthropic_key_here   # console.anthropic.com
export VOYAGE_API_KEY=your_voyage_key_here          # voyageai.com (free tier available)
# Windows: use `set` instead of `export`

# Run the server
uvicorn app.main:app --reload --port 8000
```

Then:
1. Open `http://localhost:8000` — this loads the chat UI.
2. First, ingest the sample docs: `curl -X POST http://localhost:8000/ingest`
3. Ask a question in the UI, e.g. *"How many PTO days do I get after 3 years?"*

To use your own documents, drop `.txt` or `.md` files into `/data` and
re-run the `/ingest` endpoint.

---

## Deploying on AWS Bedrock (for the AWS-hosted client case study)

To swap the LLM call from Anthropic API to AWS Bedrock:
1. Replace `Anthropic()` client in `rag_pipeline.py` with `boto3` Bedrock
   Runtime client (`bedrock-runtime`).
2. Use `bedrock.invoke_model()` with the same prompt structure, targeting
   an Anthropic Claude model available in Bedrock.
3. Optionally replace the local Chroma store with a **Bedrock Knowledge
   Base**, which manages ingestion, chunking, and retrieval natively — good
   to demonstrate for clients who specifically want an AWS-native stack.

---

## Extending this project (next steps for portfolio depth)

- [ ] Add file upload (PDF/DOCX) instead of just .txt/.md
- [ ] Add conversation memory (multi-turn follow-up questions)
- [ ] Add an eval script (RAGAS or manual test set) to measure
      answer faithfulness — strong addition for the "production skills" phase
- [ ] Add LangSmith/Langfuse tracing to show retrieval + generation steps
- [ ] Deploy live (Railway/Render/AWS) and link a demo in your case study

---

## Tech Stack
Python, FastAPI, Anthropic Claude API, Voyage AI (embeddings), ChromaDB, Docker-ready. Zero OpenAI dependency.
