"""Configuration for the evaluation "test lab" (separate from the app)."""

from pathlib import Path

# The four quality metrics, mirroring the project brief:
#   Generation: faithfulness (no hallucination), answer_correctness.
#   Retrieval:  context_precision (signal/noise), context_recall (coverage).
METRIC_NAMES: list[str] = [
    "faithfulness",
    "answer_correctness",
    "context_precision",
    "context_recall",
]

# --- Reranker evaluation (Stage 3) ---------------------------------------
# Each config: retrieve a broad set (retriever_k), then let the cross-encoder
# keep the best few (reranker_n). The reranker model itself is reused from the
# app config (src.config.RERANKER_MODEL_NAME), so the lab and the app agree.
RERANKER_CONFIGS: list[dict[str, int]] = [
    {"retriever_k": 10, "reranker_n": 2},
    {"retriever_k": 10, "reranker_n": 5},
    {"retriever_k": 20, "reranker_n": 5},
]

# --- Query rewriting evaluation (Stage 4) --------------------------------
# HyDE is tested on top of the best reranker setting. Update this with the
# winner from Stage 3 before running Stage 4.
BEST_RERANKER_STRATEGY: dict[str, int] = {"retriever_k": 10, "reranker_n": 5}

# --- Paths ---------------------------------------------------------------
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results"
