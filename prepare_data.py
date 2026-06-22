"""Bootstraps the knowledge base for the Leipzig expert.

Downloads a curated set of German Wikipedia articles about Leipzig (city,
history, landmarks, culture, people) as plain-text files into ``data/``.
Run it once: ``python prepare_data.py``. It skips articles that are already
present, so re-running only fetches the missing ones. Requests are throttled
and retried with backoff to stay within Wikipedia's rate limits.

Wikipedia text is licensed CC BY-SA 4.0. Each saved file keeps a header with
the article title, source URL and retrieval date so the knowledge base stays
attributable. This script is the "Select and Prepare a Knowledge Base" step of
the project, kept in the repo so the corpus is reproducible.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import httpx

# Curated Leipzig corpus: city overview, history, landmarks, culture, people.
ARTICLES: list[str] = [
    "Leipzig",
    "Geschichte der Stadt Leipzig",
    "Völkerschlacht bei Leipzig",
    "Völkerschlachtdenkmal",
    "Thomaskirche (Leipzig)",
    "Thomanerchor",
    "Nikolaikirche (Leipzig)",
    "Friedliche Revolution (Leipzig)",
    "Friedliche Revolution in der DDR",
    "Universität Leipzig",
    "Gewandhausorchester",
    "Auerbachs Keller",
    "Augustusplatz",
    "Leipzig Hauptbahnhof",
    "Zoo Leipzig",
    "Leipziger Buchmesse",
    "Leipziger Messe",
    "Johann Sebastian Bach",
    "Richard Wagner",
    "Felix Mendelssohn Bartholdy",
    "Leipziger Baumwollspinnerei",
    "Karl-Heine-Kanal",
    "Leipziger Neuseenland",
]

API_URL = "https://de.wikipedia.org/w/api.php"
USER_AGENT = "rag_project-leipzig/1.0 (educational WBS Coding project)"
DATA_PATH = Path(__file__).parent / "data"

# Be polite to Wikipedia: pause between requests, retry on rate limits.
DELAY_BETWEEN_REQUESTS = 1.5
MAX_RETRIES = 4


def slugify(title: str) -> str:
    keep = (c if c.isalnum() else "_" for c in title.lower())
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str] | None:
    """Returns (resolved_title, plain_text) or None if the article is missing."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        response = client.get(API_URL, params=params, timeout=30.0)
        if response.status_code == 429 and attempt < MAX_RETRIES:
            wait = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s
            print(f"    rate limited, waiting {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        break

    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        return None
    return page["title"], page["extract"]


def main() -> None:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    retrieved = date.today().isoformat()
    written = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for title in ARTICLES:
            out_path = DATA_PATH / f"leipzig_{slugify(title)}.txt"
            if out_path.exists():
                print(f"  = {title}: already present, skipped")
                written += 1
                continue

            try:
                result = fetch_extract(client, title)
            except httpx.HTTPError as error:
                print(f"  ! {title}: request failed ({error})")
                continue

            if result is None:
                print(f"  ! {title}: not found, skipped")
                continue

            resolved_title, text = result
            # A redirect can resolve to a different title than requested, so the
            # real filename uses the resolved title. Re-check it here so a
            # re-run does not rewrite an already-present (redirected) article.
            out_path = DATA_PATH / f"leipzig_{slugify(resolved_title)}.txt"
            if out_path.exists():
                print(f"  = {title} -> {resolved_title}: already present")
                written += 1
                continue

            source_url = (
                "https://de.wikipedia.org/wiki/"
                + resolved_title.replace(" ", "_")
            )
            header = (
                f"Titel: {resolved_title}\n"
                f"Quelle: {source_url}\n"
                f"Lizenz: CC BY-SA 4.0 (Wikipedia)\n"
                f"Abgerufen: {retrieved}\n"
                f"{'-' * 60}\n\n"
            )
            out_path = DATA_PATH / f"leipzig_{slugify(resolved_title)}.txt"
            out_path.write_text(header + text, encoding="utf-8")
            written += 1
            words = len(text.split())
            print(f"  + {resolved_title}  ({words} words) -> {out_path.name}")
            time.sleep(DELAY_BETWEEN_REQUESTS)  # be polite to the API

    print(f"\nDone. {written}/{len(ARTICLES)} articles saved to {DATA_PATH}.")


if __name__ == "__main__":
    main()
