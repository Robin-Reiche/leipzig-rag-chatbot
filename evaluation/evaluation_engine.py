"""Orchestrates a baseline evaluation of the Leipzig RAG system.

It runs the RAG pipeline over the evaluation questions, then uses an LLM judge
to score each answer on the four metrics and writes a CSV scorecard.
"""

import pandas as pd
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine

from evaluation.evaluation_helper_functions import (
    answer_questions,
    get_evaluation_data,
    save_results,
    score_answer_correctness,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
)
from evaluation.evaluation_model_loader import initialise_judge_llm
from src.config import SIMILARITY_TOP_K
from src.engine import get_vector_store
from src.model_loader import get_embedding_model, initialise_llm


def evaluate_baseline() -> None:
    """Evaluates the RAG system using the settings from src/config.py."""

    print("--- Stage 1: Evaluating baseline configuration ---")

    # Build the same RAG pipeline the app uses.
    llm = initialise_llm()
    embed_model = get_embedding_model()
    index: VectorStoreIndex = get_vector_store(embed_model)
    query_engine: BaseQueryEngine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        llm=llm,
    )

    questions, ground_truths = get_evaluation_data()

    print("\n--- Generating answers ---")
    answers, contexts = answer_questions(query_engine, questions)

    print("\n--- Judging answers (faithfulness, correctness, retrieval) ---")
    judge = initialise_judge_llm()
    rows: list[dict] = []
    for i, question in enumerate(questions):
        print(f"  Scoring {i + 1}/{len(questions)} …")
        rows.append(
            {
                "question": question,
                "answer": answers[i],
                "ground_truth": ground_truths[i],
                "faithfulness": score_faithfulness(judge, answers[i], contexts[i]),
                "answer_correctness": score_answer_correctness(
                    judge, answers[i], ground_truths[i]
                ),
                "context_precision": score_context_precision(
                    judge, question, contexts[i]
                ),
                "context_recall": score_context_recall(
                    judge, ground_truths[i], contexts[i]
                ),
            }
        )

    results_df = pd.DataFrame(rows)
    save_results(results_df, "baseline_evaluation")
    print("\n--- Baseline evaluation complete ---")
