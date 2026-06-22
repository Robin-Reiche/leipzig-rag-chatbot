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

# --- Paths ---------------------------------------------------------------
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results"
