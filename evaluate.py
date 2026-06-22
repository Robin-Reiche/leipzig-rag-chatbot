"""Entry point for the evaluation "test lab".

Measures the RAG system's quality with a small LLM-as-judge over four metrics.
The course lesson uses ragas, but the current ragas and langchain releases have
an incompatible API here, so this is a dependency-light re-implementation of the
same idea.

Install the base app dependencies (requirements.txt) plus the eval extra:
    uv pip install -r requirements-eval.txt
Then run:
    python evaluate.py

By default the judge uses the same LLM backend as the app (Ollama). Switching
LLM_BACKEND=groq moves both the app and the judge to Groq, so a truly
independent judge would need its own backend setting.
"""

from evaluation.evaluation_engine import evaluate_baseline

if __name__ == "__main__":
    evaluate_baseline()
