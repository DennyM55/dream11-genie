# Dream11 Genie

Educational fantasy cricket XI generator using **illustrative local sample data**, not live cricket statistics or betting advice.

## Run locally

Use Python 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Set `GROQ_API_KEY` securely in your environment or copy `.env.example` to `.env` and enter your key locally. Never commit credentials. `GROQ_MODEL` is optional; Simple demo defaults to `openai/gpt-oss-20b`.

Simple demo sends one compact request with local sample evidence. It requests plain JSON without provider-side schema enforcement or tool calling, parses and validates locally, and deterministically repairs rule violations. Empty, malformed, truncated, or unusable responses produce an error. Requests are not automatically retried. Agent learning mode preserves the LangGraph tool workflow and may consume several requests; use Simple demo for the portfolio.

## Verify

```sh
python test_offline.py
python -m pip check
```

The offline suite uses mocks only for provider responses and a noncredential test string for client construction. It checks JSON parsing, rule repair, request counts, rate limits, and Streamlit rendering/error messages without network calls. It does **not** establish live Groq compatibility.

For a live check with a real key, call `agent.pick_team_demo('CSK vs MI')` once, then assert `len(team.players) == 11` and `not validate.validate_team(team)[0]`. Do not repeat failed requests without approval. Live verification is currently blocked by the absence of a configured key.

## Streamlit Community Cloud

1. Deploy this repository with entrypoint `app.py` and Python 3.12. Dependencies come from `requirements.txt`.
2. In the app's secret settings, set `GROQ_API_KEY = "your-real-key"` and optionally `GROQ_MODEL = "openai/gpt-oss-20b"`. Do not put keys in source code.
3. Keep Simple demo selected. Verify one generation produces a valid 11-player XI, captain, vice-captain, roles, credits, and strategy before announcing the deployment as working.

No database or additional service is required. Hosting credentials are shared by visitors, so public use can exhaust your Groq quota; the UI reports provider limits without retrying.
