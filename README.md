# Leipzig Expert: a RAG Chatbot

A chatbot that knows one city well. It takes a general language model and turns it into a reliable specialist for Leipzig, its history, culture, landmarks and famous people, by grounding every answer in a knowledge base of German Wikipedia articles. The web UI shows live what the bot is doing for each question: which sources it found and how the answer forms token by token.

## Project Overview

A general LLM will happily answer questions about Leipzig, but it also invents details when it is unsure. This project fixes that with Retrieval-Augmented Generation (RAG). Instead of trusting the model's memory, every answer is built from text retrieved from a curated knowledge base, and the system prompt forces the model to say when something is not in the sources rather than guess.

The result is a focused, honest expert that stays on the facts and points back to where each answer came from.

## How It Works

RAG in three steps:

1. **Indexing** (once): the documents in `data/` are split into overlapping chunks, turned into vectors by an embedding model and stored in a persistent vector store on disk.
2. **Retrieval**: a follow-up is first condensed against the chat history into a standalone query, then a broad set of candidate chunks is pulled from the vector store and a cross-encoder reranker keeps only the most relevant few.
3. **Generation**: those chunks and the question go to the LLM, which writes an answer grounded in the sources. A low temperature and a strict system prompt keep it factual.

## Features

- **Local first**: runs fully offline on a local LLM through Ollama, no API key and no rate limits. A Groq cloud backend is available through a single config switch.
- **Streaming web UI**: a FastAPI app with Server-Sent Events shows the live status, the retrieved sources (title, similarity score, snippet and a clickable Wikipedia link) and the answer appearing token by token.
- **Model picker**: switch the local chat model at runtime from the UI, with no restart. Only the generator changes, so the conversation and sources stay intact.
- **Two-stage retrieval**: a broad vector search followed by a cross-encoder reranker that re-scores the candidates and keeps only the most relevant, for a cleaner context with less noise.
- **Follow-up handling**: questions are condensed against the chat history into a standalone query, so "and when did he die?" still retrieves the right sources.
- **Grounded answers with sources**: the system prompt ties answers to the retrieved context and tells the model to admit when the documents do not cover a question.
- **Conversation memory**: a token-budgeted memory buffer keeps follow-up questions coherent without overflowing the model's context window.
- **Multilingual embeddings**: the knowledge base is German, so it uses a multilingual embedding model instead of an English one that retrieves German text poorly.
- **Command-line mode**: a terminal chat for quick testing without the browser.
- **Reproducible knowledge base**: a script rebuilds the corpus from Wikipedia on demand, with source and license headers per file.
- **Evaluation lab and model benchmark**: a separate harness scores answer quality with an LLM judge on four metrics, plus a benchmark that compares whole models on quality and speed.

## Tech Stack

- **RAG framework**: LlamaIndex (indexing, retrieval, reranking, chat memory)
- **LLM**: local Ollama (`gemma4:e2b`) by default, switchable at runtime, Groq (`llama-3.3-70b-versatile`) optional
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`, local on CPU
- **Reranker**: `BAAI/bge-reranker-base` cross-encoder, multilingual, local on CPU
- **Web**: FastAPI, Server-Sent Events, vanilla HTML, CSS and JS (no build step)
- **Data**: German Wikipedia via the MediaWiki API (httpx)
- **Evaluation**: a custom LLM-as-judge over four metrics, pandas for the scorecard

## Project Structure

```
leipzig-rag-chatbot/
├── main.py                 entry point: starts the web UI (or the CLI)
├── prepare_data.py         downloads the Leipzig knowledge base from Wikipedia
├── evaluate.py             entry point for the evaluation lab
├── benchmark_llms.py       compares several local models on quality and speed
├── requirements.txt        app dependencies
├── requirements-eval.txt   extra dependencies for evaluation
├── data/                   the knowledge base (Wikipedia text, CC BY-SA 4.0)
├── local_storage/          vector store and embedding cache (generated, gitignored)
├── src/
│   ├── config.py           every setting in one place
│   ├── model_loader.py     builds the LLM and the embedding model
│   ├── engine.py           the RAG pipeline: indexing, retrieval, streaming chatbot
│   └── server.py           FastAPI server with SSE streaming
├── web/                    front-end (index.html, style.css, app.js)
└── evaluation/             the evaluation lab, separate from the app
```

## Setup and Usage

Prerequisites:

- [Ollama](https://ollama.com) installed and running, with the model pulled:
  ```bash
  ollama pull gemma4:e2b
  ```
- A Python 3.11 environment with the dependencies (shown here with `uv`):
  ```bash
  uv venv --python 3.11
  uv pip install -r requirements.txt
  ```

Run it:

```bash
# 1. Fetch the knowledge base once (writes 23 articles to data/)
python prepare_data.py

# 2. Start the app. The first run builds the vector store, later runs load it.
python main.py
```

Then open http://localhost:8000 and ask a question. For a terminal chat instead of the browser, run `python main.py cli`.

To use the Groq cloud backend, set `GROQ_API_KEY` in `.env` and start with `LLM_BACKEND=groq`.

## Evaluation

Rather than judging by feel, the lab in `evaluation/` measures quality with an LLM-as-judge on four metrics:

- **faithfulness**: is the answer supported by the retrieved context, with no hallucination?
- **answer correctness**: does the answer match the reference answer?
- **context precision**: how many of the retrieved chunks are relevant?
- **context recall**: does the retrieved context hold everything the answer needs?

```bash
uv pip install -r requirements-eval.txt
python evaluate.py            # baseline (default)
python evaluate.py rerank     # compare reranker settings (retrieve-k vs rerank-n)
python evaluate.py rewrite    # test HyDE query rewriting
```

Results are written as a CSV scorecard to `evaluation/evaluation_results/`. The questions and reference answers live in `evaluation/evaluation_questions.py`.

To compare whole models instead of pipeline settings, run `python benchmark_llms.py`. It sends every candidate model through the same retrieval and reranking pipeline and scores them on quality and speed with one fixed judge, so they are graded on equal footing.

## Use Your Own Topic

The pipeline is not tied to Leipzig:

1. Replace the files in `data/` with your own `.txt`, `.pdf` or `.md`.
2. Delete `local_storage/` to force a fresh index.
3. Adjust the system prompt in `src/config.py` to your topic.
4. Restart. The first run rebuilds the index.

## Data and License

The knowledge base is built from German Wikipedia articles (CC BY-SA 4.0). Each file in `data/` carries a header with its title, source URL and retrieval date, and `prepare_data.py` keeps the corpus reproducible.

## Limitations and Next Steps

This is a single-user demo: the server keeps one shared conversation and memory, which is fine for a local showcase but not for concurrent users. Retrieval already runs two stages (vector search plus a cross-encoder reranker) and condenses follow-ups against the chat history, and the evaluation lab compares reranker and query-rewriting (HyDE) settings on the four metrics. The natural next steps: a stronger, fixed evaluation judge (the lab currently grades with the active chat model, which is light), a larger question set for sharper numbers, and adding HyDE to the live app once the evaluation shows it helps.
