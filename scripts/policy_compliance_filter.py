"""Run policy compliance checks for scenario queries using OpenRouter models.

This script reads configuration from a YAML file, loads policies and verified
queries, calls an OpenRouter model to classify each query as ALLOW or DENY, and
writes the results to JSONL files in a format similar to
``results/response_results/*.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from tqdm import tqdm

# Local utilities
from utils.json_utils import read_json, read_jsonl, write_jsonl
from utils.openrouter_api_utils import create_openrouter_response
from utils.string_utils import json_style_str_to_dict


# ---------------------------------------------------------------------------
# Path helpers and global constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "scripts" / "config" / "policy_compliance_filter.yaml"

_WORKER_SETTINGS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to the repository root unless it is absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def format_policy_section(section: Dict[str, str]) -> str:
    """Converts a policy section dictionary into a bullet list string."""
    lines = []
    for key, description in section.items():
        lines.append(f"- {key}: {description}")
    return "\n".join(lines)


def build_prompt(template: str, company_name: str, allowlist: Dict[str, str], denylist: Dict[str, str], user_query: str) -> str:
    """Fill the prompt template with company-specific policy data and the query."""
    allowlist_block = format_policy_section(allowlist)
    denylist_block = format_policy_section(denylist)
    return template.format(
        company_name=company_name,
        allowlist=allowlist_block,
        denylist=denylist_block,
        user_query=user_query,
    )


def sanitize_model_name(model: str) -> str:
    """Convert a model name to a filesystem-friendly representation."""
    return model.replace("/", "_").replace(":", "-").replace(" ", "-")


def load_policies(policies_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all company policies from JSON files."""
    policies: Dict[str, Dict[str, Any]] = {}
    for path in sorted(policies_dir.glob("*.json")):
        company = path.stem
        policies[company] = read_json(str(path))
    return policies


def load_queries(query_dir: Path, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load queries for each company from JSONL files."""
    queries: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(query_dir.glob("*.jsonl")):
        company = path.stem
        records = read_jsonl(str(path))
        if limit is not None:
            records = records[:limit]
        queries[company] = records
    return queries


def init_worker(settings: Dict[str, Any]) -> None:
    """Initializer for worker processes; stores shared settings globally."""
    global _WORKER_SETTINGS
    _WORKER_SETTINGS = settings


def call_model(messages: List[Dict[str, str]]) -> str:
    """Call the OpenRouter model using shared worker settings."""
    settings = _WORKER_SETTINGS
    return create_openrouter_response(
        model=settings["model"],
        prompt_input=messages,
        max_completion_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        return_type="string",
    )


def parse_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse the model response into a dictionary."""
    try:
        return json_style_str_to_dict(response_text)
    except Exception:
        return None


def run_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute filtering for a single query."""
    settings = _WORKER_SETTINGS
    prompt = build_prompt(
        template=settings["prompt_template"],
        company_name=task["company"],
        allowlist=task["allowlist"],
        denylist=task["denylist"],
        user_query=task["query_text"],
    )

    messages: List[Dict[str, str]] = []
    system_message = settings.get("system_message")
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response_text: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    for attempt in range(1, settings["max_trials"] + 1):
        try:
            response_text = call_model(messages)
        except Exception as exc:  # API failure
            error_message = f"API error on attempt {attempt}: {exc}"
            if attempt == settings["max_trials"]:
                break
            time.sleep(settings["delay_seconds"])
            continue

        parsed = parse_response(response_text)
        if parsed is not None:
            break

        error_message = f"Parse error on attempt {attempt}: response not valid JSON"
        if attempt == settings["max_trials"]:
            break
        time.sleep(settings["delay_seconds"])

    record = task["record"]
    result: Dict[str, Any] = {
        "id": record.get("id"),
        "company": task["company"],
        "query_type": task["query_type"],
        "model": settings["model"],
        "category": record.get("category"),
        "policy": record.get("policy"),
        "query": task["query_text"],
        "response": response_text,
    }

    decision = parsed.get("decision") if parsed else None
    category = record.get("category")
    is_correct: Optional[bool] = None

    if decision and category:
        decision_upper = decision.upper()
        category_lower = category.lower()
        if decision_upper in {"ALLOW", "DENY"} and category_lower in {"allowlist", "denylist"}:
            is_correct = (decision_upper == "ALLOW" and category_lower == "allowlist") or (
                decision_upper == "DENY" and category_lower == "denylist"
            )

    if parsed is not None:
        result.update(
            {
                "decision": parsed.get("decision"),
                "confidence": parsed.get("confidence"),
                "matched_policy": parsed.get("matched_policy"),
                "reason": parsed.get("reason"),
                "is_correct": is_correct,
                "raw_result": parsed,
            }
        )
    else:
        result.update({"error": error_message, "is_correct": is_correct})

    return result


def gather_tasks(
    policies: Dict[str, Dict[str, Any]],
    query_bundles: Dict[str, Dict[str, List[Dict[str, Any]]]],
    query_field_map: Dict[str, str],
    selected_companies: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Create task dictionaries for all (company, query_type) pairs."""
    selected_set = set(c for c in (selected_companies or [])) or None
    tasks: List[Dict[str, Any]] = []

    for query_type, company_records in query_bundles.items():
        text_field = query_field_map.get(query_type)
        if text_field is None:
            continue
        for company, records in company_records.items():
            if selected_set is not None and company not in selected_set:
                continue
            policy_data = policies.get(company)
            if not policy_data:
                # Skip companies without policies
                continue
            allowlist = policy_data.get("allowlist", {})
            denylist = policy_data.get("denylist", {})
            for record in records:
                query_text = record.get(text_field)
                if not query_text:
                    continue
                tasks.append(
                    {
                        "company": company,
                        "query_type": query_type,
                        "record": record,
                        "query_text": query_text,
                        "allowlist": allowlist,
                        "denylist": denylist,
                    }
                )
    return tasks


def write_results_per_split(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Path]:
    """Write results grouped by (query_type, company) to JSONL files."""
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for item in results:
        key = (item.get("query_type", "unknown"), item.get("company", "unknown"))
        grouped.setdefault(key, []).append(item)

    written_files: List[Path] = []
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for (query_type, company), items in grouped.items():
        if not items:
            continue
        model_name = sanitize_model_name(items[0].get("model", "unknown"))
        filename = f"{query_type}_{company}_{model_name}_{timestamp}.jsonl"
        output_path = output_dir / filename
        # Ensure deterministic ordering by id then policy
        items.sort(key=lambda x: (x.get("id") or "", x.get("policy") or ""))
        write_jsonl(items, str(output_path))
        written_files.append(output_path)

    return written_files


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy compliance filtering via OpenRouter models.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to YAML configuration file.")
    parser.add_argument("--model", type=str, default=None, help="OpenRouter model name to use.")
    parser.add_argument("--n_proc", type=int, default=1, help="Number of parallel worker processes.")
    parser.add_argument("--query_types", nargs="*", help="Subset of query types to process (e.g., base allowed_edge denied_edge).")
    parser.add_argument("--companies", nargs="*", help="Optional list of company names to include.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of queries per company per type (for debugging).")
    parser.add_argument("--output_dir", type=str, default=None, help="Override the output directory from the config file.")
    parser.add_argument("--quiet", action="store_true", help="Reduce CLI output.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    config_path = resolve_path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)

    openrouter_cfg = config.get("openrouter", {})
    supported_models = openrouter_cfg.get("supported_models", [])
    default_model = supported_models[0] if supported_models else None
    model_name = args.model or default_model
    if not model_name:
        print("Model must be specified via --model or in config supported_models.", file=sys.stderr)
        return 1
    if supported_models and model_name not in supported_models:
        print(f"Warning: model '{model_name}' not in supported_models list.")

    dataset_cfg = config.get("datasets", {})
    policies_dir = resolve_path(dataset_cfg.get("policies_dir", "scenario/policies"))
    queries_dir_root = resolve_path(dataset_cfg.get("queries_dir", "scenario/queries"))
    query_types_dir = dataset_cfg.get("query_types_dir", {})

    if not policies_dir.exists():
        print(f"Policies directory not found: {policies_dir}", file=sys.stderr)
        return 1
    if not queries_dir_root.exists():
        print(f"Queries directory not found: {queries_dir_root}", file=sys.stderr)
        return 1

    retry_cfg = config.get("retry", {})
    output_cfg = config.get("output", {})

    results_dir = resolve_path(args.output_dir or output_cfg.get("results_dir", "results/filter_results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    query_types = list(query_types_dir.keys())
    if args.query_types:
        missing = [qt for qt in args.query_types if qt not in query_types_dir]
        if missing:
            print(f"Unknown query_types requested: {missing}", file=sys.stderr)
            return 1
        query_types = args.query_types

    temperature = openrouter_cfg.get("temperature", 0.1)
    max_tokens = openrouter_cfg.get("max_tokens", 500)
    max_trials = int(retry_cfg.get("max_trials", 3))
    delay_seconds = float(retry_cfg.get("delay_seconds", 1.0))

    policies = load_policies(policies_dir)

    query_field_map = {
        "base": "base_query",
        "allowed_edge": "allowed_edge_query",
        "denied_edge": "denied_edge_query",
    }

    query_bundles: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for query_type in query_types:
        dir_name = query_types_dir.get(query_type)
        if not dir_name:
            continue
        query_dir = queries_dir_root / dir_name
        if not query_dir.exists():
            print(f"Query directory missing for type '{query_type}': {query_dir}", file=sys.stderr)
            continue
        query_bundles[query_type] = load_queries(query_dir, limit=args.limit)

    if not query_bundles:
        print("No queries found to process.", file=sys.stderr)
        return 1

    selected_companies = args.companies if args.companies else None

    tasks = gather_tasks(
        policies=policies,
        query_bundles=query_bundles,
        query_field_map=query_field_map,
        selected_companies=selected_companies,
    )

    if not tasks:
        print("No tasks to process after applying filters.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded {len(tasks)} queries across {len(query_bundles)} query types.")

    settings = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_trials": max_trials,
        "delay_seconds": delay_seconds,
        "prompt_template": config.get("prompt_template", ""),
        "system_message": config.get("system_message"),
    }

    global _WORKER_SETTINGS
    _WORKER_SETTINGS = settings

    use_multiprocessing = args.n_proc and args.n_proc > 1
    results: List[Dict[str, Any]] = []

    if use_multiprocessing:
        from multiprocessing import Pool

        with Pool(processes=args.n_proc, initializer=init_worker, initargs=(settings,)) as pool:
            for item in tqdm(pool.imap_unordered(run_single_task, tasks), total=len(tasks)):
                results.append(item)
    else:
        # Single-process fallback
        for task in tqdm(tasks):
            results.append(run_single_task(task))

    written_files = write_results_per_split(results, results_dir)

    if not args.quiet:
        print("Written result files:")
        for path in written_files:
            print(f"  - {path.relative_to(ROOT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
