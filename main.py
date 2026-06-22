"""Entry point for the Leipzig RAG expert.

Default: launch the web UI at http://localhost:8000
CLI mode: `python main.py cli` for a terminal chat (no browser).
"""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from src.engine import main_chat_loop

        main_chat_loop()
        return

    import uvicorn

    print("--- Leipzig-Experte ---")
    print("Web-UI startet auf http://localhost:8000  (Strg+C zum Beenden)")
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
