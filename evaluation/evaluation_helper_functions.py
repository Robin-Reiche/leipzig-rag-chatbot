"""Helpers for the LLM-as-judge evaluation pipeline."""

import re
from datetime import datetime

import pandas as pd
from httpx import HTTPError
from llama_index.core.llms import LLM
from llama_index.core.query_engine import BaseQueryEngine

from evaluation.evaluation_config import EVALUATION_RESULTS_PATH, METRIC_NAMES
from evaluation.evaluation_questions import EVALUATION_DATA


def get_evaluation_data() -> tuple[list[str], list[str]]:
    """Splits EVALUATION_DATA into parallel lists of questions and truths."""

    questions = [item["question"] for item in EVALUATION_DATA]
    ground_truths = [item["ground_truth"] for item in EVALUATION_DATA]
    return questions, ground_truths


def answer_questions(
    query_engine: BaseQueryEngine,
    questions: list[str],
) -> tuple[list[str], list[list[str]]]:
    """Runs the RAG system over each question, collecting answers + contexts."""

    answers: list[str] = []
    contexts: list[list[str]] = []
    for i, question in enumerate(questions):
        print(f"  Answering {i + 1}/{len(questions)}: {question[:50]}…")
        try:
            response = query_engine.query(question)
            answers.append(str(response).strip())
            contexts.append(
                [node.get_content() for node in response.source_nodes]
            )
        except (HTTPError, TimeoutError) as error:
            print(f"    ! generation failed ({error}), recording empty answer")
            answers.append("")
            contexts.append([])
    return answers, contexts


# --- LLM-as-judge scoring -----------------------------------------------
def _parse_score(text: str) -> float:
    """Extracts the first number from the judge's reply and clamps to [0, 1]."""

    match = re.search(r"\d+(?:[.,]\d+)?", text)
    if not match:
        return 0.0
    value = float(match.group(0).replace(",", "."))
    return max(0.0, min(1.0, value))


def _judge(llm: LLM, prompt: str) -> float:
    """Asks the judge for a score. Returns NaN if the call fails (e.g. timeout),
    so one slow call records as missing instead of crashing the whole run."""

    try:
        return _parse_score(str(llm.complete(prompt)))
    except (HTTPError, TimeoutError) as error:
        print(f"    ! judge call failed ({error}), recording NaN")
        return float("nan")


def score_faithfulness(llm: LLM, answer: str, contexts: list[str]) -> float:
    """How well the answer is supported by the retrieved context (it must not invent facts)."""

    context = "\n\n".join(contexts)
    prompt = (
        "Du bist ein strenger Prüfer. Bewerte, ob die ANTWORT vollständig "
        "durch den KONTEXT gestützt wird.\n"
        "1.0 = jede Aussage der Antwort steht im Kontext. "
        "0.0 = die Antwort erfindet Fakten oder widerspricht dem Kontext.\n"
        "Antworte ausschließlich mit einer Zahl zwischen 0 und 1.\n\n"
        f"KONTEXT:\n{context}\n\nANTWORT:\n{answer}\n\nScore:"
    )
    return _judge(llm, prompt)


def score_answer_correctness(
    llm: LLM, answer: str, ground_truth: str
) -> float:
    """How well the answer matches the factual ground truth."""

    prompt = (
        "Vergleiche die ANTWORT mit der MUSTERLÖSUNG.\n"
        "1.0 = inhaltlich vollständig korrekt und passend. "
        "0.0 = inhaltlich falsch.\n"
        "Wenn die Musterlösung besagt, dass eine Information nicht vorhanden "
        "ist, ist eine ehrliche 'weiß ich nicht'-Antwort korrekt (1.0).\n"
        "Antworte ausschließlich mit einer Zahl zwischen 0 und 1.\n\n"
        f"MUSTERLÖSUNG:\n{ground_truth}\n\nANTWORT:\n{answer}\n\nScore:"
    )
    return _judge(llm, prompt)


def score_context_precision(
    llm: LLM, question: str, contexts: list[str]
) -> float:
    """Fraction of retrieved chunks that are actually relevant to the question.

    Done in a single judge call (the judge sees all chunks and estimates the
    relevant fraction) to keep the local evaluation fast and robust.
    """

    if not contexts:
        return 0.0
    numbered = "\n\n".join(
        f"ABSCHNITT {i + 1}:\n{chunk}" for i, chunk in enumerate(contexts)
    )
    prompt = (
        f"Es wurden {len(contexts)} Abschnitte abgerufen, um die FRAGE zu "
        "beantworten. Schätze, welcher ANTEIL davon tatsächlich relevant ist.\n"
        "1.0 = alle Abschnitte sind relevant. 0.0 = keiner ist relevant.\n"
        "Antworte ausschließlich mit einer Zahl zwischen 0 und 1.\n\n"
        f"FRAGE:\n{question}\n\n{numbered}\n\nAnteil relevant:"
    )
    return _judge(llm, prompt)


def score_context_recall(
    llm: LLM, ground_truth: str, contexts: list[str]
) -> float:
    """Whether the retrieved context contains what the ground truth needs."""

    context = "\n\n".join(contexts)
    prompt = (
        "Enthält der KONTEXT die Informationen, die nötig sind, um die "
        "MUSTERLÖSUNG zu belegen?\n"
        "1.0 = alle nötigen Informationen sind im Kontext enthalten. "
        "0.0 = die nötigen Informationen fehlen.\n"
        "Antworte ausschließlich mit einer Zahl zwischen 0 und 1.\n\n"
        f"MUSTERLÖSUNG:\n{ground_truth}\n\nKONTEXT:\n{context}\n\nScore:"
    )
    return _judge(llm, prompt)


def score_rows(
    judge: LLM,
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
    contexts: list[list[str]],
) -> list[dict]:
    """Scores each answer on the four metrics with the judge LLM."""

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
    return rows


def save_results(
    results_df: pd.DataFrame,
    filename_prefix: str,
    group_by: list[str] | None = None,
) -> None:
    """Saves a detailed per-question CSV and an averaged summary CSV.

    Without ``group_by`` the summary is a single averaged column. With it (used
    by the multi-config stages) the metrics are averaged per configuration, so
    each chunking/reranker/HyDE setting gets its own row to compare.
    """

    EVALUATION_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    detailed_path = (
        EVALUATION_RESULTS_PATH / f"{filename_prefix}_detailed_{timestamp}.csv"
    )
    results_df.to_csv(detailed_path, index=False)
    print(f"\nDetailed results: {detailed_path}")

    if group_by:
        summary = results_df.groupby(group_by)[METRIC_NAMES].mean().round(3)
    else:
        summary = (
            results_df[METRIC_NAMES].mean().round(3).to_frame("average_score")
        )
    summary_path = (
        EVALUATION_RESULTS_PATH / f"{filename_prefix}_summary_{timestamp}.csv"
    )
    summary.to_csv(summary_path)
    print(f"Summary of average scores: {summary_path}\n")
    print(summary.to_string())
