"""Central configuration for the Leipzig RAG expert.

Everything that controls the pipeline lives here so behaviour can be changed in
one place: which LLM backend to use, the embedding model, chunking, retrieval,
memory and storage paths.
"""

import os
from pathlib import Path

# --- LLM backend ---------------------------------------------------------
# "ollama" runs a local model (default, no API key, no rate limits).
# "groq" uses the Groq cloud API (set GROQ_API_KEY in .env). Switch with the
# LLM_BACKEND environment variable, e.g. LLM_BACKEND=groq python main.py
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama").lower()

# Local Ollama model. gemma4:e2b fully fits the 8 GB GPU and is far faster than
# gemma4:12b (which spills to CPU and times out on long answers here), at equal
# quality on the eval questions. Swap with `ollama pull <tag>` then change here.
OLLAMA_MODEL: str = "gemma4:e2b"
# Groq cloud model, used only when LLM_BACKEND=groq.
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# A local 12B model can take ~2 minutes for a long, non-streamed answer, so the
# timeout is generous. The web UI streams, so users see tokens long before this.
LLM_REQUEST_TIMEOUT: float = 300.0
# Low temperature keeps answers factual and grounded in the sources.
LLM_TEMPERATURE: float = 0.1
# Context window for the local Ollama model. Must comfortably fit the system
# prompt + retrieved chunks + conversation history + the generated answer,
# otherwise Ollama silently truncates the prompt and answer quality drops.
OLLAMA_CONTEXT_WINDOW: int = 8192

# --- System prompt: the grounded Leipzig expert persona ------------------
# This is the single most important lever for "sticking to the facts". It tells
# the model to answer only from the retrieved context and to admit when it does
# not know, instead of hallucinating.
LLM_SYSTEM_PROMPT: str = (
    "Du bist ein sachkundiger, freundlicher Experte für die Stadt Leipzig – "
    "ihre Geschichte, Sehenswürdigkeiten, Kultur und berühmten "
    "Persönlichkeiten.\n\n"
    "Beantworte Fragen ausschließlich auf Basis des bereitgestellten Kontexts "
    "aus der Wissensbasis. Wenn die Antwort dort nicht enthalten ist, sage "
    "offen, dass du es anhand der vorliegenden Dokumente nicht beantworten "
    "kannst, und erfinde nichts.\n\n"
    "Antworte auf Deutsch, klar und präzise. Beantworte die Frage vollständig "
    "und gehe auf die wesentlichen im Kontext genannten Punkte ein, statt nur "
    "ein einzelnes Detail herauszugreifen. Nenne wo möglich konkrete Fakten wie "
    "Namen, Jahreszahlen und Orte aus den Quellen, ohne abzuschweifen."
)

# --- Embedding model -----------------------------------------------------
# The knowledge base is in German, so we use a multilingual model. The English
# all-MiniLM-L6-v2 from the lesson retrieves German text poorly. This one is
# small, fast on CPU and handles German well. (This is exactly the "which
# embedding model fits my data?" question from the project brief.)
EMBEDDING_MODEL_NAME: str = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- RAG / vector store --------------------------------------------------
# How many of the most relevant chunks to retrieve per question. 6 gives the
# generator enough context to answer completely without drowning it in noise.
SIMILARITY_TOP_K: int = 6
# Chunk size and overlap (in tokens) when splitting documents.
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# --- Reranker ------------------------------------------------------------
# A second, more precise stage after the vector search: retrieve a broad set,
# then let a cross-encoder re-score and keep only the best few. This hands the
# generator a cleaner, higher-signal context. Turn off with RERANK_ENABLED=false.
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
# Multilingual cross-encoder. The knowledge base is German, so an English-only
# reranker (e.g. ms-marco-MiniLM) would score German pairs poorly, the same
# reason the embedding model above is multilingual. Downloaded once on first run.
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
# Broad first-stage retrieval, then rerank down to the top N for the generator.
RERANK_RETRIEVE_TOP_K: int = 12
RERANKER_TOP_N: int = 4

# --- Query condensing ----------------------------------------------------
# On a follow-up ("Und wann wurde er geboren?") the bare question retrieves
# poorly. When there is chat history, first rewrite the question into a stand-
# alone one (one extra LLM call per turn, only when history exists). Turn off
# with CONDENSE_ENABLED=false.
CONDENSE_ENABLED: bool = os.getenv("CONDENSE_ENABLED", "true").lower() == "true"
CONDENSE_PROMPT_TEMPLATE: str = (
    "Formuliere die FOLGEFRAGE zu einer eigenständigen, vollständigen Frage um, "
    "die ohne den bisherigen Gesprächsverlauf verständlich ist. Löse dabei "
    "Bezüge wie 'er', 'sie', 'dort', 'das' anhand des Verlaufs auf. Gib "
    "ausschließlich die umformulierte Frage zurück, ohne Vorrede.\n\n"
    "GESPRÄCHSVERLAUF:\n{chat_history}\n\n"
    "FOLGEFRAGE:\n{question}\n\n"
    "Eigenständige Frage:"
)

# --- Chat memory ---------------------------------------------------------
# Total token budget for the prompt minus the answer: system prompt + injected
# context + conversation history must fit in this. Sized to leave room for the
# answer inside OLLAMA_CONTEXT_WINDOW.
CHAT_MEMORY_TOKEN_LIMIT: int = 6000

# --- Persistent storage paths --------------------------------------------
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage" / "embedding_model"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage" / "vector_store"
