from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import LANGUAGES, settings
from .rag import Pipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipe"] = await run_in_threadpool(Pipeline)
    yield


app = FastAPI(title="Kisan Sahayak - Farmer Advisory API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(default="", max_length=5000)
    lang: str = "en"
    top_k: int | None = Field(default=None, ge=1, le=6)


@app.get("/api/health")
def health():
    p: Pipeline = state["pipe"]
    return {"ok": True, "model": p.llm.name, "chunks": len(p.retriever.chunks),
            "translator": settings.translator}


@app.get("/api/languages")
def languages():
    p: Pipeline = state["pipe"]
    return [{"code": c, "name": n, "enabled": p.supported(c)} for c, n in LANGUAGES.items()]


@app.post("/api/ask")
async def ask(req: AskRequest):
    p: Pipeline = state["pipe"]
    if req.lang not in LANGUAGES or not p.supported(req.lang):
        raise HTTPException(400, f"Language '{req.lang}' is not enabled (set TRANSLATOR=indictrans2).")
    return await run_in_threadpool(p.ask, req.question, req.lang, req.top_k)


@app.post("/api/reindex")
async def reindex():
    p: Pipeline = state["pipe"]
    await run_in_threadpool(p.retriever.build)
    return {"chunks": len(p.retriever.chunks)}


# Frontend is plain static files, no build step.
app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
