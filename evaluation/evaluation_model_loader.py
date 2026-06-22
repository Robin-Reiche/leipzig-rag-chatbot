"""The evaluation "parts store": builds the judge LLM.

By default this reuses the same backend as the app (Ollama, local). The judge
reads an answer plus its context/ground truth and returns a numeric score.

Note: ideally the judge is a *different* model from the one being tested, so it
does not mark its own homework. With a local-only setup we reuse the same model
for simplicity; switch LLM_BACKEND=groq to grade with a different model.
"""

from llama_index.core.llms import LLM

from src.model_loader import initialise_llm


def initialise_judge_llm() -> LLM:
    """Returns the LLM used to grade the RAG system's answers."""

    return initialise_llm()
