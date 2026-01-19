  

<div align="center">
  <h1>COMPASS: A Framework for Policy Alignment Evaluation</h1>
    
  [![arXiv](https://img.shields.io/badge/arXiv-2601.01836-b31b1b.svg)](https://arxiv.org/abs/2601.01836)
  <a href="https://huggingface.co/collections/AIM-Intelligence/compass" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face-yellow.svg></a>

</div>

**COMPASS** is a framework for evaluating **policy alignment**: given only an organization’s policy (e.g., allow/deny rules), it enables you to benchmark whether an LLM’s responses comply with that policy in structured, enterprise-like scenarios.

This repository provides tools to:
1.  **Define a custom policy** for your organization.
2.  **Generate a benchmark** of synthetic queries (standard and adversarial) tailored to that policy.
3.  **Evaluate LLMs** on how well they adhere to your rules.

For reproducing the experiments from the paper (including RAG scenarios and full-scale results), please see [REPRODUCE.md](REPRODUCE.md).

## 🚀 Quick Start

### 1. Installation

```bash
conda create -n compass python=3.11
conda activate compass
pip install -r requirements.txt
```

Set up your API keys in `.env` (see `.env.sample`). The exact credentials you need depend on which providers/models you select in `scripts/config/*.yaml` (for synthesis, evaluation, and judging).

```bash
cp .env.sample .env
# Edit .env to add your keys
```

#### Required credentials (common)

- **OpenAI judge (default)**: `OPENAI_API_KEY` (used by `scripts/response_judge.py` unless you change the judge config)
- **OpenRouter (denied_edge synthesis, optional response generation)**: `OPENROUTER_API_KEY` (used by `scripts/denied_edge_queries_synthesis.py`)
- **Vertex / Anthropic Vertex (allowed_edge synthesis, optional)**:
  - If you use **Claude on Vertex** via `anthropic[vertex]`, set up **Google Cloud credentials** (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) and ensure your Vertex project/region are configured in the YAML files under `scripts/config/`.
  - If you use **Gemini via `google-genai`** in this repo, `VERTEX_API_KEY` is required (see `scripts/utils/vertex_api_utils.py`).

> Tip: If you are starting from scratch, you can run **base query generation + verification + response evaluation** first, and add edge/RAG workflows later.

### 2. Testbed Dataset

We provide a comprehensive testbed dataset covering 8 industry verticals (Automotive, Healthcare, Financial, etc.) generated using COMPASS. You can access the **Testbed Dataset** on Hugging Face:

👉 **[AIM-Intelligence/COMPASS-Policy-Alignment-Testbed-Dataset](https://huggingface.co/datasets/AIM-Intelligence/COMPASS-Policy-Alignment-Testbed-Dataset)**

This dataset serves as a reference for what COMPASS generates and allows you to test models immediately without generating your own data.
The **testbed queries corresponding to the verified query buckets under `scenario/queries/verified_*`** are published there (as Parquet).

---

## 🛠️ Usage: Creating a Custom Benchmark

Follow these steps to create a policy alignment benchmark for your own organization.

### Step 1: Define Your Policy, Context, and System Prompt

To build a custom benchmark and evaluate responses, you typically provide:

- **Policy** + **Context**: required for query generation.
- **System prompt**: required for response generation (evaluation).

**1. Policy File (`scenario/policies/MyOrg.json`):**
Define `allowlist` (topics you WANT to answer) and `denylist` (topics you MUST refuse).

```json
{
  "allowlist": {
    "product_support": "Technical support and usage guidelines for MyOrg's software products, including installation, debugging, and API usage.",
    "pricing": "Publicly available pricing tiers (Free, Pro, Enterprise) and feature comparison tables."
  },
  "denylist": {
    "competitors": "Comparisons with CompetitorX or CompetitorY, or market share analysis.",
    "internal_security": "Details about internal server infrastructure, employee credentials, or unpatched vulnerabilities."
  }
}
```

**2. Context File (`scenario/contexts/MyOrg.txt`):**
Provide a description of your organization to help the LLM generate realistic scenarios.

```text
MyOrg is a leading provider of cloud-based project management software...
```

**3. System Prompt File (`scenario/system_prompts/MyOrg.txt`):**
Provide the system prompt that the model will use when responding to queries. You can write any prompt you want the model to follow.

```text
You are a helpful assistant for MyOrg. You must strictly follow the company's content policies...
```

### Step 2: Generate and Verify Evaluation Queries

Use the synthesis scripts to generate user queries based on your policy, and then run verification scripts to ensure quality.

> Note: The synthesis scripts enumerate **all** `scenario/policies/*.json` files by default.
>
> **Recommended (to run a single custom org safely):** work in a separate branch/copy, and temporarily keep **only** these three files for your org:
> - `scenario/policies/MyOrg.json`
> - `scenario/contexts/MyOrg.txt`
> - `scenario/system_prompts/MyOrg.txt`
>
> This is the most reliable way to avoid accidental API calls for other scenarios.  
> You can also use `--debug/--max-companies` to limit the run, but it is less explicit than isolating the files.
>
> **New:** You can run scripts for specific companies with `--company`. Example:
> `python scripts/base_queries_synthesis.py --company MyOrg`

**1. Generate Standard Queries (Base):**
```bash
python scripts/base_queries_synthesis.py
```
*This generates standard questions for both allowlist and denylist topics.*
*To run a specific company (or multiple):*  
`python scripts/base_queries_synthesis.py --company MyOrg OtherOrg`

**2. Verify Base Queries:**
```bash
python scripts/base_queries_verification.py
```
*This validates the generated queries and saves the approved ones to `scenario/queries/verified_base/`.*
*To run a specific company (or multiple):*  
`python scripts/base_queries_verification.py --company MyOrg OtherOrg`

**3. Generate Edge Cases (Adversarial/Borderline):**
```bash
python scripts/allowed_edge_queries_synthesis.py
python scripts/denied_edge_queries_synthesis.py
```
*   `allowed_edge`: Tricky questions that *seem* risky but should be answered.
*   `denied_edge`: Adversarial attacks (jailbreaks, social engineering) trying to elicit denied info.
*To run a specific company (or multiple):*  
`python scripts/allowed_edge_queries_synthesis.py --company MyOrg OtherOrg`  
`python scripts/denied_edge_queries_synthesis.py --company MyOrg OtherOrg`

Prerequisites:
- `allowed_edge_queries_synthesis.py` uses **Vertex** utilities (see `scripts/utils/vertex_api_utils.py`), so make sure your Vertex/GCP auth is configured.
- `denied_edge_queries_synthesis.py` calls **OpenRouter**, so you need `OPENROUTER_API_KEY`.

**4. Verify Edge Cases:**
```bash
python scripts/allowed_edge_queries_verification.py
python scripts/denied_edge_queries_verification.py
```
*Validated queries are saved to `scenario/queries/verified_allowed_edge/` and `scenario/queries/verified_denied_edge/`.*
*To run a specific company (or multiple):*  
`python scripts/allowed_edge_queries_verification.py --company MyOrg OtherOrg`  
`python scripts/denied_edge_queries_verification.py --company MyOrg OtherOrg`

### Step 3: Run Evaluation

1.  **Generate Responses**: Run your target LLM against the generated queries.
    You must specify the model, company, and query type. The script will automatically load the verified queries.

    ```bash
    # Example for OpenRouter models
    python scripts/response_generation_openrouter.py \
      --model "openai/gpt-4-turbo" \
      --company "MyOrg" \
      --query_type "base"
    ```
    *(Run separately for `base`, `allowed_edge`, and `denied_edge`)*

2.  **Judge Compliance**: Use an LLM-as-a-Judge to score the responses.
    ```bash
    python scripts/response_judge.py "response_results" -n 5
    ```

3.  **Analyze Results**:
    ```bash
    python scripts/analyze_judged_results.py --target-directory judge_results
    ```

## Project Structure

- `scenario/`: Your input data (policies, contexts) and generated benchmarks.
  - `policies/`: **Put your JSON policy here.**
  - `contexts/`: **Put your company description TXT here.**
  - `system_prompts/`: **Put your system prompt TXT here.**
  - `queries/`: Generated benchmark data.
- `scripts/`: Tools for synthesis and evaluation.
- `results/`: Output from model runs and evaluations.

## Citation

If you use COMPASS in your research, please cite:

```bibtex
@misc{choi2026compass,
      title={COMPASS: A Framework for Evaluating Organization-Specific Policy Alignment in LLMs}, 
      author={Dasol Choi and DongGeon Lee and Brigitta Jesica Kartono and Helena Berndt and Taeyoun Kwon and Joonwon Jang and Haon Park and Hwanjo Yu and Minsuk Kahng},
      year={2026},
      eprint={2601.01836},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.01836}, 
}
```
