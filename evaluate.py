"""Entry point for the evaluation "test lab".

Measures the RAG system's quality with a small LLM-as-judge over four metrics.
The course lesson uses ragas, but the current ragas and langchain releases have
an incompatible API here, so this is a dependency-light re-implementation of the
same idea.

Install the base app dependencies (requirements.txt) plus the eval extra:
    uv pip install -r requirements-eval.txt
Then run a stage:
    python evaluate.py            # baseline (default)
    python evaluate.py rerank     # Stage 3: reranker configs
    python evaluate.py rewrite    # Stage 4: HyDE query rewriting
    python evaluate.py all        # every stage in order

By default the judge uses the same LLM backend as the app (Ollama). Switching
LLM_BACKEND=groq moves both the app and the judge to Groq, so a truly
independent judge would need its own backend setting.
"""

import sys

from evaluation.evaluation_engine import (
    evaluate_baseline,
    evaluate_query_rewriting,
    evaluate_reranker_strategies,
)

STAGES = {
    "baseline": evaluate_baseline,
    "rerank": evaluate_reranker_strategies,
    "rewrite": evaluate_query_rewriting,
}

if __name__ == "__main__":
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "baseline"

    if choice == "all":
        for stage in STAGES.values():
            stage()
    elif choice in STAGES:
        STAGES[choice]()
    else:
        options = ", ".join([*STAGES, "all"])
        print(f"Unknown stage '{choice}'. Choose from: {options}")
        sys.exit(1)
