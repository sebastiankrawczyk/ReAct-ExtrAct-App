# ReAct‑ExtrAct Runner 

ReAct‑ExtrAct Runner is a Streamlit application for source‑grounded data extraction from scientific PDFs. It builds vector indexes over your papers, retrieves the most relevant snippets, and synthesizes concise answers for user‑defined extraction fields. You can optionally provide categorical Codes to get structured outputs (e.g., algorithm lists).

## Features
- Streamlit UI with a guided, 4‑step workflow
- Three extraction modes:
  - Naive RAG: single‑pass retrieval and answer (fast)
  - Iterative RAG: two‑step refinement (balanced)
  - ReAct‑ExtrAct: planning, confidence scoring, per‑group meeting, and a final consistency sweep (thorough)
- Codes (categorical options) per field for concise, structured answers
- Planner heuristics override from the UI (optional)
- Per‑run artifacts and CSV export

## Prerequisites
- Python 3.10+
- pip

Optional, depending on how you run models and parsing:
- Hosted APIs: OpenRouter or OpenAI (LLM), Llama Cloud (parsing)
- Local stack: Ollama (LLM) and GROBID (PDF parsing)

## Installation
```bash
pip install -r requirements.txt
```

## Running the UI
```bash
streamlit run frontend/app_new.py
```
Then open http://localhost:8501.

## Input, Configuration, and Keys
1) Place PDFs in `input/` (or upload via the UI).

2) Define extraction fields and optional Codes in the UI, then click “Save Fields”. This writes to `config/queries.py` for reproducibility.

3) API keys and providers:
- Provide keys on the Settings page or via environment variables/.env:
  - `OPENROUTER_API_KEY` or `OPENAI_API_KEY`
  - `LLAMA_CLOUD_API_KEY` (for Llama Cloud parsing)
  - `COHERE_API_KEY` (optional)
- For local runs: set provider(s) to Ollama in Step 3 and configure GROBID (URL and toggle) if using local parsing.

## Using the UI (Workflow)
The New Extraction page walks you through:
- Demonstration Video (optional)
- Step 1 — Setup Project: name your project and upload PDFs
- Step 2 — Define Extraction Fields: enter topics and optional Codes
- Step 3 — Configure & Run:
  - Pick provider(s): OpenRouter/OpenAI or Ollama
  - Optionally toggle GROBID (local parser) and check its status
  - Select the execution model:
    - “OpenRouter Exec Model” is shown only when provider is OpenRouter
    - “Ollama Exec Model” is used when provider is Ollama
  - Planner heuristic (ReAct‑ExtrAct only): leave empty to use defaults; add guidance to override
  - Start extraction
- Step 4 — Examine Results: browse answers, open Inspector (🔍), and export CSV

## Planner Heuristics (ReAct‑ExtrAct)
- Default planner guidance lives in `utils/react_agent_utils.py` as `PLANNER_HEURISTICS`.
- In the UI, “Planner heuristic (ReAct‑ExtrAct)” overrides the default when non‑empty.
- Environment overrides are also supported:
  - `PLANNER_HEURISTICS_OVERRIDE` (full replacement)
  - or enable `HUMAN_HEURISTIC=1` with `HUMAN_HEURISTIC_TEXT` to append to defaults
If empty, the system uses the default heuristics.

## Outputs
- `output/<timestamp_mode>__/` per run, with per‑paper JSON artifacts:
  - `<paper>_result.json`: native outputs
  - `<paper>_baseline_like.json`: normalized list of answers for easy tabulation
- CSV export available from the UI (choose papers, fields, optional columns)

## Storage and Performance
- Vector indexes are stored under `storage/` to speed up subsequent runs.
- You can clear `output/` and `storage/` between demos to start fresh.

## Troubleshooting
- Missing keys: ensure required API keys are set (Settings page or env vars).
- Llama Cloud parsing: requires `LLAMA_CLOUD_API_KEY`.
- GROBID (local): verify the URL and that the service is running; use the “Check GROBID status” button.
- Provider/model errors: double‑check selected provider matches configured keys and models.

## Quick Start (TL;DR)
1) `pip install -r requirements.txt`
2) `streamlit run frontend/app_new.py`
3) Upload PDFs, define fields, pick a mode, configure provider/keys, run
4) Inspect answers and export CSV

---
Notes:
- This repository prioritizes transparent, source‑grounded extraction rather than pure end‑to‑end generation.
- Planner sub‑queries are optional and not required for the default ReAct‑ExtrAct flow.

## Deployment

### Streamlit Community Cloud (recommended for demo)
1) Fork this repository.
2) Create a new app in Streamlit Cloud and point it to `frontend/app_new.py`.
3) In “Secrets”, provide any keys you plan to use (leave blank for a free demo if your workspace already grants access):
   - `OPENROUTER_API_KEY`
   - `OPENAI_API_KEY`
   - `LLAMA_CLOUD_API_KEY`
   - `COHERE_API_KEY` (optional)
4) Optionally set environment variables in the app (Advanced settings), for example default providers/models if desired.
5) Deploy. Open the URL and use the UI to upload PDFs, define fields, and run.

Notes:
- The public demo typically disables local components (Ollama/GROBID). Use hosted APIs on Cloud.
- For larger PDFs or many files, consider increasing app resources or running locally.

### Local (production‑like)
1) Install dependencies: `pip install -r requirements.txt`
2) Start the UI: `streamlit run frontend/app_new.py`
3) For local LLMs, install and run Ollama; for local parsing, run a GROBID server and set its URL in Step 3 → Parsing Options.
4) Provide keys via environment or the Settings page if using hosted providers.
