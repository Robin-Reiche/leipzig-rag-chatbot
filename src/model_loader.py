"""The "parts store": builds the low-level components the app needs.

Its only job is to create configured model objects (the LLM and the embedding
model). It knows about API keys and model names so the rest of the app does not.
"""

import os

from dotenv import load_dotenv
from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL,
    LLM_BACKEND,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_MODEL,
)

# Load environment variables from the .env file (needed for the Groq backend).
load_dotenv()


def initialise_llm() -> LLM:
    """Builds the generator LLM for the configured backend (ollama or groq)."""

    if LLM_BACKEND == "groq":
        from llama_index.llms.groq import Groq

        api_key: str | None = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_BACKEND is 'groq' but GROQ_API_KEY is not set. "
                "Add it to your .env file or switch to LLM_BACKEND=ollama."
            )
        return Groq(
            api_key=api_key,
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
        )

    # Default: local Ollama. Requires the Ollama app running and the model
    # pulled (`ollama pull gemma4:12b`).
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=OLLAMA_MODEL,
        request_timeout=LLM_REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        context_window=OLLAMA_CONTEXT_WINDOW,
    )


def get_embedding_model() -> HuggingFaceEmbedding:
    """Builds the embedding model that turns text into searchable vectors."""

    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )
