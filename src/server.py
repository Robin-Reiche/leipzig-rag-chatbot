"""Web server for the Leipzig RAG expert.

A small FastAPI app that serves the chat UI and streams answers over
Server-Sent Events (SSE). The stream lets the front-end show what the chatbot
is doing in real time: searching the knowledge base, which sources it found,
and the answer appearing token by token.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from llama_index.core.schema import NodeWithScore

from src.config import DATA_PATH, GROQ_MODEL, LLM_BACKEND, OLLAMA_MODEL
from src.engine import RagChatbot, build_streaming_chatbot
from src.model_loader import initialise_llm

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

WEB_DIR: Path = Path(__file__).parent.parent / "web"

# A single shared chatbot. This is a single-user demo, so one conversation and
# one memory buffer is fine. Use the "Neue Unterhaltung" button (/api/reset) to
# start fresh. ``current_model`` tracks the active generator for the UI picker.
_state: dict[str, Any] = {
    "chatbot": None,
    "sources_meta": {},
    "current_model": GROQ_MODEL if LLM_BACKEND == "groq" else OLLAMA_MODEL,
}

_STOP = object()  # sentinel marking the end of the token stream


def _load_source_metadata() -> dict[str, dict[str, str]]:
    """Reads the 'Titel:' and 'Quelle:' header of each knowledge-base file.

    Lets the UI show a readable title and a clickable source link per result.
    """

    meta: dict[str, dict[str, str]] = {}
    for path in DATA_PATH.glob("*.txt"):
        title, source = path.stem, ""
        with path.open(encoding="utf-8") as handle:
            for _ in range(6):  # header is in the first few lines
                line = handle.readline()
                if line.startswith("Titel:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("Quelle:"):
                    source = line.split(":", 1)[1].strip()
        meta[path.name] = {"title": title, "source": source}
    return meta


def _format_sources(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    """Turns retrieved nodes into compact, UI-friendly source cards."""

    sources_meta: dict[str, dict[str, str]] = _state["sources_meta"]
    sources: list[dict[str, Any]] = []
    for node_with_score in nodes:
        node = node_with_score.node
        file_name = node.metadata.get("file_name", "")
        info = sources_meta.get(file_name, {"title": file_name, "source": ""})
        full_text = " ".join(node.get_content().split())
        snippet = full_text[:240]
        sources.append(
            {
                "title": info["title"],
                "source": info["source"],
                "score": round(float(node_with_score.score or 0.0), 3),
                "snippet": snippet + ("…" if len(full_text) > 240 else ""),
            }
        )
    return sources


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Builds the RAG engine once when the server starts."""

    print("--- Starting Leipzig RAG server ---")
    _state["sources_meta"] = _load_source_metadata()
    # Building the chatbot downloads the embedding model and the vector store on
    # first run, so do it off the event loop to keep startup responsive.
    _state["chatbot"] = await asyncio.to_thread(build_streaming_chatbot)
    print("--- Server ready at http://localhost:8000 ---")
    yield


app = FastAPI(title="Leipzig RAG Expert", lifespan=lifespan)


def _sse(payload: dict[str, Any]) -> str:
    """Formats a payload as a Server-Sent Event."""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _safe_next(generator: Iterator[str]) -> Any:
    """next() that returns the _STOP sentinel instead of raising."""

    try:
        return next(generator)
    except StopIteration:
        return _STOP


async def _chat_events(message: str) -> AsyncIterator[str]:
    """Drives one question through the pipeline, emitting SSE events."""

    chatbot: RagChatbot = _state["chatbot"]

    # Step 1+2: retrieval (blocking, so run it off the event loop).
    yield _sse({"type": "status", "stage": "retrieving",
                "text": "Durchsuche die Wissensbasis …"})
    nodes: list[NodeWithScore] = await asyncio.to_thread(
        chatbot.retrieve, message
    )

    yield _sse({"type": "sources", "sources": _format_sources(nodes)})
    yield _sse({"type": "status", "stage": "generating",
                "text": "Formuliere eine Antwort aus den Quellen …"})

    # Step 3: stream the answer token by token without blocking the event loop.
    token_gen: Iterator[str] = chatbot.stream_answer(message, nodes)
    while True:
        token = await asyncio.to_thread(_safe_next, token_gen)
        if token is _STOP:
            break
        yield _sse({"type": "token", "text": token})

    yield _sse({"type": "done"})


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    """Streams the chatbot's reasoning steps and answer for one question."""

    body = await request.json()
    message = (body.get("message") or "").strip()

    async def stream() -> AsyncIterator[str]:
        if not message:
            yield _sse({"type": "error", "text": "Leere Frage."})
            return
        try:
            async for event in _chat_events(message):
                yield event
        except Exception as error:  # surface failures to the UI instead of hanging
            yield _sse({"type": "error", "text": f"Fehler: {error}"})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset() -> dict[str, str]:
    """Clears the conversation memory to start a fresh chat."""

    chatbot: RagChatbot | None = _state["chatbot"]
    if chatbot is not None:
        chatbot.reset()
    return {"status": "ok"}


@app.get("/api/info")
async def info() -> dict[str, Any]:
    """Backend details shown in the UI header."""

    doc_count = len(list(DATA_PATH.glob("*.txt")))
    return {
        "backend": LLM_BACKEND,
        "model": _state["current_model"],
        "documents": doc_count,
    }


def _list_ollama_models() -> list[str]:
    """Lists the chat models pulled into the local Ollama, sorted by name."""

    try:
        response = httpx.get(OLLAMA_TAGS_URL, timeout=5.0)
        response.raise_for_status()
        names = [m["name"] for m in response.json().get("models", [])]
    except Exception:  # Ollama down or unreachable: return nothing, UI copes
        return []
    return sorted(names)


@app.get("/api/models")
async def models() -> dict[str, Any]:
    """Lists models for the picker. Empty/locked for the Groq backend."""

    if LLM_BACKEND == "groq":
        return {"models": [GROQ_MODEL], "current": GROQ_MODEL, "backend": "groq"}

    available = await asyncio.to_thread(_list_ollama_models)
    current = _state["current_model"]
    if current and current not in available:
        available = [current, *available]
    return {"models": available, "current": current, "backend": LLM_BACKEND}


@app.post("/api/model")
async def set_model(request: Request) -> dict[str, Any]:
    """Switches the chat generator to another local Ollama model at runtime."""

    if LLM_BACKEND == "groq":
        return {"status": "error", "message": "Modellwahl nur im Ollama-Backend."}

    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        return {"status": "error", "message": "Kein Modell angegeben."}

    # Reject models that are not installed. The Ollama client itself only fails
    # at query time, which would leave the chatbot pointing at a broken model.
    # (Skip the check if the tags call failed, so a transient hiccup does not
    # block a legitimate switch.)
    available = await asyncio.to_thread(_list_ollama_models)
    if available and model not in available:
        return {
            "status": "error",
            "message": f"Modell '{model}' ist nicht installiert.",
        }

    chatbot: RagChatbot | None = _state["chatbot"]
    if chatbot is None:
        return {"status": "error", "message": "Chatbot ist noch nicht bereit."}

    try:
        # Building the client is cheap; the model loads into VRAM on the next
        # question. Run off the event loop to stay consistent and safe.
        llm = await asyncio.to_thread(initialise_llm, model)
    except Exception as error:  # surface a bad model name to the UI
        return {"status": "error", "message": f"Fehler: {error}"}

    chatbot.set_llm(llm)
    _state["current_model"] = model
    return {"status": "ok", "model": model}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# Serve the static front-end assets (CSS, JS). Registered last so it does not
# shadow the API routes above.
app.mount("/", StaticFiles(directory=WEB_DIR.as_posix()), name="static")
