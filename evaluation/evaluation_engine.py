"""Orchestrates a baseline evaluation of the Leipzig RAG system.

It runs the RAG pipeline over the evaluation questions, then uses an LLM judge
to score each answer on the four metrics and writes a CSV scorecard.
"""

import pandas as pd
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
    TransformQueryEngine,
)

from evaluation.evaluation_config import (
    BEST_RERANKER_STRATEGY,
    RERANKER_CONFIGS,
)
from evaluation.evaluation_helper_functions import (
    answer_questions,
    get_evaluation_data,
    save_results,
    score_answer_correctness,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
    score_rows,
)
from evaluation.evaluation_model_loader import initialise_judge_llm
from src.config import RERANKER_MODEL_NAME, SIMILARITY_TOP_K
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


def evaluate_reranker_strategies() -> None:
    """Stage 3: tests reranker settings on top of the current chunking.

    For each (retriever_k, reranker_n) config it retrieves a broad set, reranks
    it with the cross-encoder, then judges the answers. The summary is grouped
    per config so you can see which retrieve/rerank split scores best.
    """

    print("\n--- Stage 3: Evaluating reranker strategies ---")

    llm = initialise_llm()
    embed_model = get_embedding_model()
    index: VectorStoreIndex = get_vector_store(embed_model)
    judge = initialise_judge_llm()
    questions, ground_truths = get_evaluation_data()

    all_rows: list[dict] = []
    for config in RERANKER_CONFIGS:
        retriever_k, reranker_n = config["retriever_k"], config["reranker_n"]
        print(
            f"\n--- Reranker config: retrieve_k={retriever_k}, "
            f"rerank_n={reranker_n} ---"
        )

        retriever = index.as_retriever(similarity_top_k=retriever_k)
        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL_NAME, top_n=reranker_n
        )
        query_engine: BaseQueryEngine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm,
        )

        answers, contexts = answer_questions(query_engine, questions)
        rows = score_rows(judge, questions, answers, ground_truths, contexts)
        for row in rows:
            row["retriever_k"] = retriever_k
            row["reranker_n"] = reranker_n
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    save_results(
        results_df,
        "reranker_evaluation",
        group_by=["retriever_k", "reranker_n"],
    )
    print("\n--- Reranker evaluation complete ---")


def evaluate_query_rewriting() -> None:
    """Stage 4: tests HyDE query rewriting on top of the best reranker setting.

    Runs the best reranked pipeline twice, once plain and once with HyDE (the
    LLM first drafts a hypothetical answer and retrieves with it). The grouped
    summary shows whether HyDE helps, especially context_recall.
    """

    print("\n--- Stage 4: Evaluating query rewriting (HyDE) ---")

    llm = initialise_llm()
    embed_model = get_embedding_model()
    index: VectorStoreIndex = get_vector_store(embed_model)
    judge = initialise_judge_llm()
    questions, ground_truths = get_evaluation_data()

    retriever_k = BEST_RERANKER_STRATEGY["retriever_k"]
    reranker_n = BEST_RERANKER_STRATEGY["reranker_n"]

    all_rows: list[dict] = []
    for use_hyde in [False, True]:
        print(f"\n--- HyDE enabled: {use_hyde} ---")

        retriever = index.as_retriever(similarity_top_k=retriever_k)
        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL_NAME, top_n=reranker_n
        )
        base_engine: BaseQueryEngine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm,
        )

        if use_hyde:
            hyde = HyDEQueryTransform(llm=llm, include_original=True)
            query_engine: BaseQueryEngine = TransformQueryEngine(
                base_engine, query_transform=hyde
            )
        else:
            query_engine = base_engine

        answers, contexts = answer_questions(query_engine, questions)
        rows = score_rows(judge, questions, answers, ground_truths, contexts)
        for row in rows:
            row["use_hyde"] = use_hyde
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    save_results(
        results_df, "query_rewrite_evaluation", group_by=["use_hyde"]
    )
    print("\n--- Query rewrite evaluation complete ---")
