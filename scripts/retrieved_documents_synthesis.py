import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool, Manager
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

from utils.anthropic_api_utils import create_response_chat
from utils.json_utils import write_json
from utils.string_utils import json_style_str_to_dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCENARIO_DIR = os.path.join(PROJECT_ROOT, "scenario")
CONTEXT_DIR = os.path.join(SCENARIO_DIR, "contexts")
QUERY_DIR = os.path.join(SCENARIO_DIR, "queries")

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "retrieved_documents_synthesis.yaml")

QUERY_BUCKETS = {
    "verified_base": "base_query",
    "verified_allowed_edge": "allowed_edge_query",
    "verified_denied_edge": "denied_edge_query",
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_company_context(company: str) -> str:
    context_path = os.path.join(CONTEXT_DIR, f"{company}.txt")
    if not os.path.exists(context_path):
        raise FileNotFoundError(
            f"Context file for company '{company}' not found at {context_path}"
        )
    with open(context_path, "r", encoding="utf-8") as f:
        return f.read()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            entries.append(json.loads(line))
    return entries


def build_prompt(
    template: str,
    company_context: str,
    query_bucket: str,
    query_record: Dict[str, Any],
    query_text: str,
    documents_per_query: int,
) -> str:
    replacements = {
        "company_context": company_context,
        "company_name": query_record.get("company", ""),
        "query_bucket": query_bucket,
        "query_id": query_record.get("id", ""),
        "policy": query_record.get("policy", ""),
        "category": query_record.get("category", ""),
        "query_text": query_text,
        "documents_per_query": str(documents_per_query),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(f"{{{placeholder}}}", value if value is not None else "")
    return prompt


def validate_retrieved_documents(
    parsed: Any,
    expected_count: int,
) -> Tuple[bool, Optional[List[Dict[str, Any]]]]:
    if not isinstance(parsed, dict):
        return False, None
    docs = parsed.get("retrieved_documents")
    if not isinstance(docs, list):
        return False, None
    if len(docs) < expected_count:
        return False, None
    for doc in docs:
        if not isinstance(doc, dict):
            return False, None
        for field in ("doc_id", "title", "document_source", "passage", "relevance_explanation"):
            if field not in doc or not isinstance(doc[field], str) or not doc[field].strip():
                return False, None
    return True, docs


def call_anthropic_with_retry(
    model: str,
    temperature: float,
    max_tokens: int,
    messages: List[Dict[str, str]],
    max_trials: int,
    backoff_seconds: float,
) -> str:
    last_error: Optional[Exception] = None
    delay = backoff_seconds
    for trial in range(1, max_trials + 1):
        try:
            return create_response_chat(
                model=model,
                prompt_input=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                return_type="string",
            )
        except Exception as error:
            last_error = error
            print(
                f"    ⚠️ Anthropic API error on trial {trial}/{max_trials}: {error}"
            )
            if trial < max_trials:
                time.sleep(delay)
                delay *= 1.5
    raise RuntimeError(f"Anthropic API failed after {max_trials} attempts") from last_error


def save_retrieved_documents(
    output_dir: str,
    query_id: str,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{query_id}.json")
    write_json(payload, output_path)


def process_query(
    args: Tuple[
        Dict[str, Any],
        str,
        str,
        Dict[str, Any],
        str,
        str,
        bool,
    ],
    lock: Optional[Manager().Lock] = None,
) -> None:
    (
        config,
        company_context,
        query_bucket,
        query_record,
        query_text,
        output_dir,
        overwrite,
    ) = args
    
    query_id = query_record.get("id")
    output_path = os.path.join(output_dir, f"{query_id}.json")
    if not overwrite and os.path.exists(output_path):
        tqdm.write(f"    ✓ {query_id} already processed, skipping")
        return

    documents_per_query = config["generation"]["documents_per_query"]
    prompt = build_prompt(
        template=config["prompt_template"],
        company_context=company_context,
        query_bucket=query_bucket,
        query_record=query_record,
        query_text=query_text,
        documents_per_query=documents_per_query,
    )
    messages = [{"role": "user", "content": prompt}]

    max_trials = config["retry"]["max_trials"]
    backoff_seconds = config["retry"]["backoff_seconds"]
    model = config["anthropic"]["model"]
    temperature = config["anthropic"]["temperature"]
    max_tokens = config["anthropic"]["max_tokens"]

    for trial in range(1, max_trials + 1):
        tqdm.write(f"    🔄 Generating retrieved docs (trial {trial}/{max_trials}) for {query_id}")
        try:
            response = call_anthropic_with_retry(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                max_trials=1,
                backoff_seconds=backoff_seconds,
            )
        except Exception as error:
            tqdm.write(f"    ✗ API call failed: {error}")
            if trial == max_trials:
                raise
            time.sleep(backoff_seconds)
            continue

        try:
            parsed = json_style_str_to_dict(response)
        except Exception as parse_error:
            tqdm.write(f"    ⚠️ JSON parsing failed for {query_id}: {parse_error}")
            if trial == max_trials:
                payload = {
                    "query": query_record,
                    "query_text": query_text,
                    "query_bucket": query_bucket,
                    "model": model,
                    "raw_response": response,
                    "parse_error": str(parse_error),
                }
                if lock:
                    with lock:
                        save_retrieved_documents(output_dir, query_id, payload)
                else:
                    save_retrieved_documents(output_dir, query_id, payload)
                return
            continue

        is_valid, docs = validate_retrieved_documents(parsed, documents_per_query)
        if not is_valid:
            tqdm.write(f"    ⚠️ Validation failed for {query_id}")
            if trial == max_trials:
                payload = {
                    "query": query_record,
                    "query_text": query_text,
                    "query_bucket": query_bucket,
                    "model": model,
                    "raw_response": parsed,
                    "validation_error": True,
                }
                if lock:
                    with lock:
                        save_retrieved_documents(output_dir, query_id, payload)
                else:
                    save_retrieved_documents(output_dir, query_id, payload)
                return
            time.sleep(backoff_seconds)
            continue

        payload = {
            "query": query_record,
            "query_text": query_text,
            "query_bucket": query_bucket,
            "model": model,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retrieved_documents": docs[:documents_per_query],
        }
        if lock:
            with lock:
                save_retrieved_documents(output_dir, query_id, payload)
        else:
            save_retrieved_documents(output_dir, query_id, payload)
        tqdm.write(f"    ✓ Saved retrieved docs for {query_id}")
        return


def discover_companies(buckets: List[str]) -> List[str]:
    companies = set()
    for bucket in buckets:
        bucket_dir = os.path.join(QUERY_DIR, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        for file_name in os.listdir(bucket_dir):
            if file_name.endswith(".jsonl"):
                companies.add(os.path.splitext(file_name)[0])
    return sorted(companies)


def process_bucket_queries(
    config: Dict[str, Any],
    company: str,
    company_context: str,
    bucket: str,
    queries: List[Dict[str, Any]],
    output_dir: str,
    overwrite: bool,
    n_proc: int,
) -> None:
    """Process all queries for a specific company/bucket combination."""
    source_field = QUERY_BUCKETS[bucket]
    
    # Prepare arguments for each query
    args_list = []
    for query in queries:
        query_text = query.get(source_field)
        if not query_text:
            tqdm.write(f"    ⚠️ Missing '{source_field}' in query {query.get('id')}, skipping")
            continue
        args_list.append((
            config,
            company_context,
            bucket,
            query,
            query_text,
            output_dir,
            overwrite,
        ))
    
    if not args_list:
        return
    
    if n_proc == 1:
        # Sequential processing
        for args in tqdm(args_list, desc=f"Processing {company}/{bucket}", unit="query"):
            try:
                process_query(args)
            except Exception as error:
                tqdm.write(f"    ✗ Failed to process {args[3].get('id')}: {error}")
    else:
        # Parallel processing
        manager = Manager()
        lock = manager.Lock()
        
        with Pool(processes=n_proc) as pool:
            process_func = partial(process_query, lock=lock)
            results = pool.imap_unordered(process_func, args_list)
            
            for _ in tqdm(
                results,
                total=len(args_list),
                desc=f"Processing {company}/{bucket} (parallel)",
                unit="query",
            ):
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic retrieved documents for all verified query sets.",
    )
    parser.add_argument(
        "--companies",
        nargs="*",
        help="Limit processing to specific companies (default: all discovered companies).",
    )
    parser.add_argument(
        "--buckets",
        nargs="*",
        choices=list(QUERY_BUCKETS.keys()),
        help="Limit processing to specific query buckets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate retrieved documents even if output files already exist.",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=1,
        help="Number of parallel processes for API calls (default: 1).",
    )

    args = parser.parse_args()
    config = load_config(CONFIG_PATH)

    selected_buckets = args.buckets or list(QUERY_BUCKETS.keys())
    companies = args.companies or discover_companies(selected_buckets)

    if config.get("debug", {}).get("enabled"):
        max_companies = config["debug"].get("max_companies")
        max_queries = config["debug"].get("max_queries_per_company")
    else:
        max_companies = None
        max_queries = None

    if max_companies is not None:
        companies = companies[:max_companies]

    documents_subdir_name = config["generation"]["output_subdir_name"]
    os.makedirs(os.path.join(QUERY_DIR, documents_subdir_name), exist_ok=True)

    print(
        f"Processing companies: {', '.join(companies)}\n"
        f"Buckets: {', '.join(selected_buckets)}\n"
        f"Parallel processes: {args.n_proc}"
    )

    for company in companies:
        try:
            company_context = load_company_context(company)
        except FileNotFoundError as missing_ctx_error:
            print(f"⚠️ Skipping {company}: {missing_ctx_error}")
            continue

        print(f"\n=== Company: {company} ===")
        for bucket in selected_buckets:
            bucket_dir = os.path.join(QUERY_DIR, bucket)
            query_file = os.path.join(bucket_dir, f"{company}.jsonl")
            if not os.path.exists(query_file):
                print(f"  - No queries for {bucket}, skipping")
                continue

            print(f"  - Bucket: {bucket}")
            queries = read_jsonl(query_file)
            if max_queries is not None:
                queries = queries[:max_queries]

            output_dir = os.path.join(bucket_dir, documents_subdir_name, company)
            
            try:
                process_bucket_queries(
                    config=config,
                    company=company,
                    company_context=company_context,
                    bucket=bucket,
                    queries=queries,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    n_proc=args.n_proc,
                )
            except Exception as error:
                print(f"    ✗ Failed to process bucket {bucket}: {error}")


if __name__ == "__main__":
    main()

