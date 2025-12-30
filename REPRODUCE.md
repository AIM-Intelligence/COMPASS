# Reproducing COMPASS Experiments

This document contains instructions for reproducing the experimental results presented in the COMPASS paper, specifically focusing on the RAG workflows and full-scale benchmarks across all 8 provided scenarios.

## Prerequisites (credentials)

- **OpenAI (judge and some response generation)**: set `OPENAI_API_KEY`
- **OpenRouter (denied_edge synthesis and optional response generation)**: set `OPENROUTER_API_KEY`
- **Vertex / Anthropic Vertex (RAG docs and allowed_edge synthesis)**:
  - For **Claude on Vertex**: configure Google Cloud auth (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) and ensure your Vertex project/region are set appropriately in `scripts/config/*.yaml`.
  - For **Gemini via `google-genai`** in this repo: set `VERTEX_API_KEY` (see `scripts/utils/vertex_api_utils.py`).

> Note: Query synthesis and evaluation providers are configurable via YAML under `scripts/config/`. If you change the configured models/providers, you may need additional environment variables (e.g., `ANTHROPIC_API_KEY`).

## Core Workflows for Reproduction

### 1. Generate and Verify Queries
Use these scripts to build the evaluation set from the source scenarios in `scenario/`:

```bash
python scripts/base_queries_synthesis.py            # allowlist + denylist seeds
python scripts/base_queries_verification.py         # validate synthesis output
python scripts/allowed_edge_queries_synthesis.py
python scripts/allowed_edge_queries_verification.py
python scripts/denied_edge_queries_synthesis.py
python scripts/denied_edge_queries_verification.py
```
Each script reads from `scenario/` and writes JSON/JSONL files back into the same tree.

### 2. Generate Retrieved Documents (RAG)
To evaluate models in a Retrieval-Augmented Generation setting, first synthesize fictitious but contextually consistent documents for each query:
```bash
python scripts/retrieved_documents_synthesis.py --companies CityGov AutoViaMotors --buckets verified_base verified_allowed_edge verified_denied_edge --n_proc 4
```
This creates `scenario/queries/<bucket>/retrieved_documents/<company>/<query_id>.json` files containing synthetic passages aligned with company contexts. Use `--overwrite` to regenerate existing documents and `--n_proc` to control parallel processing (default: 1). Adjust `scripts/config/retrieved_documents_synthesis.yaml` to control document count, model (`claude-haiku-4-5`), and prompt template.

### 3. Generate Model Responses
Choose the client that matches your provider credentials. 

**Standard (non-RAG) response generation:**
You need to specify the model, company, and query type.

```bash
# Example: Evaluate GPT-4 on AutoViaMotors base queries
python scripts/response_generation_openai.py \
  --model "gpt-4-0613" \
  --company "AutoViaMotors" \
  --query_type "base"

# Example: Evaluate a model via OpenRouter
python scripts/response_generation_openrouter.py \
  --model "meta-llama/llama-3-70b-instruct" \
  --company "AutoViaMotors" \
  --query_type "denied_edge"
```

> Note: `response_generation_openrouter.py` supports `base`, `allowed_edge`, and `denied_edge` in code, even if the CLI help text only mentions edge types.

**RAG-enabled response generation:**
```bash
python scripts/response_generation_rag_openai.py \
  --model "gpt-4-0613" \
  --company "AutoViaMotors" \
  --query_type "allowed_edge"
```
RAG scripts load retrieved documents (if available) and inject them into the user prompt. System prompts are automatically trimmed at the `# Context` delimiter to avoid duplicating company background. If documents are missing for a query, the script logs a warning and proceeds without grounding.

Results are stored under `results/`.

### 4. Judge and Analyze Responses
Score responses against policy rubrics, then inspect summaries.
```bash
# Judge the responses (directory name usually matches the run type)
python scripts/response_judge.py "response_results" -n 10 -v

# Analyze the judged results
python scripts/analyze_judged_results.py --target-directory judge_results --pretty-model-names
```

**Filter-based Evaluation:**
```bash
# 1. Run the compliance filter simulation
python scripts/policy_compliance_filter.py --model <model> --n_proc 10 --query_types base denied_edge allowed_edge

# 2. Judge the filtered results
python scripts/filtered_judge_results.py

# 3. Analyze
python scripts/analyze_filter_and_response_results.py --run responses --pretty-model-names
python scripts/analyze_filter_and_response_results.py --run filters --pretty-model-names
```

## Configuration Notes
- YAML config in `scripts/config/` controls prompt templates, retry behaviour, and model defaults.
- Many scripts support parallelism via `--n_proc`; pick values consistent with your rate limits.
- Set provider-specific parameters (model IDs, temperature, max tokens) either in configs or command-line arguments.

## Data & Output Conventions
- Intermediate artifacts stay inside the repository to keep datasets versionable.
- Response runs append to JSONL logs; re-run scripts with caution if deterministic ordering matters.
- Verification and judge scripts emit status messages to STDOUT—redirect if you need persistent logs.

## Troubleshooting
- Check `.env` when API calls fail; most scripts rely on `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, or Google application default credentials.
- Use the `debug` section in YAML configs to reduce workload while iterating (e.g., limit companies).
- Ensure `results/` directories exist or let scripts create them automatically; they are included in `.gitignore` when necessary.
