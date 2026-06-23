"""Benchmark several local LLMs on the Leipzig RAG pipeline.

For each candidate model it runs the evaluation questions through the same
retrieval + reranking pipeline the app uses, varying only the generator LLM,
and measures:
  - quality: faithfulness and answer_correctness, scored by one fixed judge
    (gemma4:12b) so every model is graded on equal footing, not by itself.
  - speed: average answer latency and tokens per second.

Retrieval and reranking are built once and shared, so differences come from the
generator alone. Each model is warmed up once (loaded into VRAM) before timing.

Run:
    python benchmark_llms.py

Every model must be pulled first (ollama pull <model>); missing models are
skipped. Results go to evaluation/evaluation_results/llm_benchmark_*.csv and are
printed as a table ranked by quality, then speed.
"""

import re
import time
from datetime import datetime

import pandas as pd
import tiktoken
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama

from evaluation.evaluation_config import EVALUATION_RESULTS_PATH
from evaluation.evaluation_helper_functions import (
    get_evaluation_data,
    score_answer_correctness,
    score_faithfulness,
)
from src.config import (
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    OLLAMA_CONTEXT_WINDOW,
    RERANK_RETRIEVE_TOP_K,
    RERANKER_MODEL_NAME,
    RERANKER_TOP_N,
)
from src.engine import get_vector_store
from src.model_loader import get_embedding_model

# Candidate generators, with gemma4:12b as the reference. Pull each first.
BENCHMARK_MODELS: list[str] = [
    "gemma4:12b",
    "qwen3:4b",
    "qwen3:8b",
    "gemma4:e2b",
    "llama3.2:3b",
]

# One fixed judge so every model is graded by the same grader, not itself.
JUDGE_MODEL: str = "gemma4:12b"

# Thinking models (e.g. qwen3) wrap their reasoning in <think>…</think>. We strip
# it so the judge scores the actual answer, but the full latency still counts:
# the thinking time is real time-to-answer and the benchmark should show it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# Reasoning models that emit long <think> traces before the answer. Both gemma4
# and qwen3 do this by default, which inflates latency, can hit the timeout, and
# (for the judge) pollutes the numeric score. We disable thinking so the
# benchmark measures the actual answer. llama3.2 does not think.
_THINKING_MODELS: tuple[str, ...] = ("gemma4", "qwen3")


def _build_ollama(model: str) -> Ollama:
    """Builds an Ollama LLM with the same settings the app uses.

    Thinking is switched off for reasoning models (see ``_THINKING_MODELS``) so
    every model is timed and judged on its actual answer, not its reasoning
    trace. Non-reasoning models do not get the flag.
    """

    extra: dict = {}
    if model.startswith(_THINKING_MODELS):
        extra["thinking"] = False

    return Ollama(
        model=model,
        request_timeout=LLM_REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        context_window=OLLAMA_CONTEXT_WINDOW,
        **extra,
    )


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def benchmark_models() -> None:
    """Runs every benchmark model through the RAG pipeline and reports.

    Two passes, on purpose. Generation and judging use different models, and an
    8 GB GPU holds only one at a time. Interleaving them per question would evict
    the generator after every answer and reload it cold on the next, wrecking the
    latency measurement. So we generate all answers for a model while it stays
    hot (clean timings), and judge everything at the end with the judge loaded
    once.
    """

    print("--- LLM benchmark on the Leipzig RAG pipeline ---")

    embed_model = get_embedding_model()
    index: VectorStoreIndex = get_vector_store(embed_model)
    questions, ground_truths = get_evaluation_data()

    # Retrieval + rerank are identical for every model, so build them once.
    retriever = index.as_retriever(similarity_top_k=RERANK_RETRIEVE_TOP_K)
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME, top_n=RERANKER_TOP_N
    )
    encoder = tiktoken.get_encoding("cl100k_base")  # consistent token proxy

    # --- Pass 1: generation. Keep each model hot across its questions. -------
    records: list[dict] = []
    for model in BENCHMARK_MODELS:
        print(f"\n=== Generating with {model} ===")
        llm = _build_ollama(model)

        # Warm-up: load into VRAM so the first timed question is not penalised by
        # the load. Doubles as a "is this model pulled?" check.
        try:
            llm.complete("Hallo")
        except Exception as error:
            print(f"  ! skipping {model} (not available: {error})")
            continue

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm,
        )

        for i, question in enumerate(questions):
            print(f"  Q{i + 1}/{len(questions)}: {question[:50]}…")
            start = time.perf_counter()
            try:
                response = query_engine.query(question)
                answer = _strip_thinking(str(response))
                contexts = [n.get_content() for n in response.source_nodes]
            except Exception as error:
                print(f"    ! generation failed ({error})")
                answer, contexts = "", []
            elapsed = time.perf_counter() - start

            tokens = len(encoder.encode(answer))
            records.append(
                {
                    "model": model,
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truths[i],
                    "latency_s": round(elapsed, 1),
                    "tokens": tokens,
                    "tokens_per_s": round(tokens / elapsed, 1) if elapsed else 0.0,
                }
            )

    if not records:
        print("\nNo models were benchmarked. Pull at least one model first.")
        return

    # --- Pass 2: judging. Load the judge once and score every answer. --------
    print(f"\n=== Judging all answers with {JUDGE_MODEL} ===")
    judge = _build_ollama(JUDGE_MODEL)

    rows: list[dict] = []
    for record in records:
        print(f"  Scoring {record['model']} | {record['question'][:40]}…")
        rows.append(
            {
                "model": record["model"],
                "question": record["question"],
                "answer": record["answer"],
                "faithfulness": score_faithfulness(
                    judge, record["answer"], record["contexts"]
                ),
                "answer_correctness": score_answer_correctness(
                    judge, record["answer"], record["ground_truth"]
                ),
                "latency_s": record["latency_s"],
                "tokens": record["tokens"],
                "tokens_per_s": record["tokens_per_s"],
            }
        )

    _save_and_report(pd.DataFrame(rows))


def _save_and_report(df: pd.DataFrame) -> None:
    """Writes the detailed + summary CSVs and prints a ranked table."""

    EVALUATION_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    detailed_path = (
        EVALUATION_RESULTS_PATH / f"llm_benchmark_detailed_{timestamp}.csv"
    )
    df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results: {detailed_path}")

    summary = (
        df.groupby("model")
        .agg(
            faithfulness=("faithfulness", "mean"),
            answer_correctness=("answer_correctness", "mean"),
            latency_s=("latency_s", "mean"),
            tokens_per_s=("tokens_per_s", "mean"),
        )
        .round(2)
    )
    # Quality = mean of the two generator-dependent metrics (retrieval is shared).
    summary["quality"] = (
        summary[["faithfulness", "answer_correctness"]].mean(axis=1).round(2)
    )
    summary = summary.sort_values(
        ["quality", "tokens_per_s"], ascending=False
    )

    summary_path = (
        EVALUATION_RESULTS_PATH / f"llm_benchmark_summary_{timestamp}.csv"
    )
    summary.to_csv(summary_path)
    print(f"Summary: {summary_path}\n")
    print(summary.to_string())

    print(f"\nBest quality:  {summary['quality'].idxmax()}")
    print(f"Fastest:       {summary['tokens_per_s'].idxmax()}")


if __name__ == "__main__":
    benchmark_models()
