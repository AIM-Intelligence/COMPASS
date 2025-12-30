"""
 uv run python scripts/response_generation_rag_openrouter.py --model google/gemma-3-27b-it
 uv run python scripts/response_generation_rag_openrouter.py --model meta-llama/llama-3.3-70b-instruct
 uv run python scripts/response_generation_rag_openrouter.py --model qwen/qwen-2.5-72b-instruct
"""

import os
import json
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from multiprocessing import Pool, Manager
from functools import partial

import dotenv
import requests
from tqdm import tqdm
from utils.json_utils import append_jsonl

# ---------------------------------------------------------------------------
# Environment setup and constants
# ---------------------------------------------------------------------------

dotenv.load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SYSTEM_PROMPT_DIR = os.path.join(BASE_DIR, "scenario", "system_prompts")
QUERY_DIR = os.path.join(BASE_DIR, "scenario", "queries")
RESULT_DIR = os.path.join(BASE_DIR, "results", "rag_response_results")
PAYLOAD_DIR = os.path.join(BASE_DIR, "scenario", "payloads")
PAYLOAD_RESULT_DIR = os.path.join(BASE_DIR, "results", "payloads_rag_response_results")
ROBUST_RESULT_DIR = os.path.join(BASE_DIR, "results", "robust_rag_response_results")

# Create results directory if it doesn't exist
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PAYLOAD_RESULT_DIR, exist_ok=True)
os.makedirs(ROBUST_RESULT_DIR, exist_ok=True)

# Field name mapping within jsonl
QUERY_FIELD_MAP = {
    "base": "base_query",
    "allowed_edge": "allowed_edge_query",
    "denied_edge": "denied_edge_query",
}
RETRIEVED_DOC_SUBDIR = "retrieved_documents"
RAG_USER_PROMPT_TEMPLATE = """
<user_query>
{query_text}
</user_query>

<retrieved_documents>
{retrieved_documents_block}
</retrieved_documents>

Instructions:
1. Reference document evidence explicitly using bracketed doc IDs (e.g., [DOC-001]).
2. If no document fully answers the query, say so while offering the closest compliant guidance.
3. Never fabricate facts that conflict with the retrieved passages or system prompt.
"""

DEFAULT_MAX_TOKENS = 2048
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_N_PROC = 100  # Default number of parallel processes

PAYLOAD_CATEGORY_ORDER = (
    "simple_refusals",
    "complex_refusals",
    "simple_acceptances",
    "complex_acceptances",
)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_list_arg(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    normalized: List[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        normalized.extend(part for part in parts if part)
    return normalized or None


def discover_companies(query_types: List[str]) -> List[str]:
    companies: Set[str] = set()
    for query_type in query_types:
        query_dir = os.path.join(QUERY_DIR, f"verified_{query_type}")
        if not os.path.isdir(query_dir):
            continue
        for filename in os.listdir(query_dir):
            if filename.lower().endswith(".jsonl"):
                companies.add(filename[:-6])
    return sorted(companies)


def trim_system_prompt(prompt_text: str) -> str:
    delimiter = "# Context"
    if delimiter in prompt_text:
        prompt_text = prompt_text.split(delimiter)[0]
    return prompt_text.strip()


def load_system_prompt(company: str) -> tuple[str, str]:
    """Load the system prompt corresponding to the company name and return (text, path)."""
    txt_path = os.path.join(SYSTEM_PROMPT_DIR, f"{company}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_prompt = f.read().strip()
            return trim_system_prompt(raw_prompt), txt_path

    raise FileNotFoundError(
        f"System prompt for company '{company}' not found in {SYSTEM_PROMPT_DIR}."
    )


def load_queries(company: str, query_type: str) -> List[Dict]:
    """Load queries corresponding to the company and query type (allowed_edge / denied_edge)."""
    query_file = os.path.join(QUERY_DIR, f"verified_{query_type}", f"{company}.jsonl")
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")

    queries: List[Dict] = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            data = json.loads(line)
            queries.append(data)
    return queries


def call_openrouter_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    url: str | None = None,
    retries: int = 30,
) -> str:
    """OpenRouter ChatCompletion API wrapper function.

    Args:
        model: OpenRouter model name (e.g., "mistralai/mistral-large")
        messages: List of messages in OpenAI format
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens (response length)
        url: OpenRouter endpoint; defaults to the official endpoint
        retries: Number of retry attempts on failure
    Returns:
        ChatCompletion response text (assistant role)
    """

    if url is None:
        url = DEFAULT_OPENROUTER_URL

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. Please set it in .env file or as environment variable."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,  # Streaming responses are not used
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices")
                if choices and len(choices) > 0:
                    return choices[0]["message"]["content"]
                raise RuntimeError(f"No choices in response: {data}")
            raise RuntimeError(
                f"API call failed ({resp.status_code}): {resp.text[:200]}"
            )
        except Exception as e:
            if attempt < retries:
                print(f"⚠️ OpenRouter API failed {attempt}/{retries} attempts — retrying: {e}")
            else:
                raise


def load_payload_examples(company: str) -> Dict[str, Any]:
    payload_path = os.path.join(PAYLOAD_DIR, f"{company}.json")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(
            f"Payload file for company '{company}' not found at {payload_path}."
        )

    with open(payload_path, "r", encoding="utf-8") as f:
        try:
            payloads = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse payload JSON at {payload_path}: {exc}") from exc

    expected_company = payloads.get("company_name")
    if expected_company and expected_company != company:
        raise ValueError(
            f"Payload company name '{expected_company}' does not match requested company '{company}'."
        )

    examples = payloads.get("simulated_examples")
    if not isinstance(examples, dict):
        raise ValueError("Payload file is missing 'simulated_examples' dictionary.")

    for category in PAYLOAD_CATEGORY_ORDER:
        category_examples = examples.get(category)
        if not isinstance(category_examples, list) or not category_examples:
            raise ValueError(
                f"Payload category '{category}' is missing or empty for company '{company}'."
            )
        for entry in category_examples:
            if "user_query" not in entry or "assistant_response" not in entry:
                raise ValueError(
                    f"Payload entry in category '{category}' is missing required keys."
                )

    return payloads


def load_retrieved_documents(
    company: str,
    query_type: Optional[str],
    query_id: Optional[str],
) -> List[Dict[str, Any]]:
    if not company or not query_type or not query_id:
        return []

    bucket_dir = os.path.join(QUERY_DIR, f"verified_{query_type}")
    doc_path = os.path.join(
        bucket_dir,
        RETRIEVED_DOC_SUBDIR,
        company,
        f"{query_id}.json",
    )
    if not os.path.exists(doc_path):
        return []

    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    docs = payload.get("retrieved_documents")
    return docs if isinstance(docs, list) else []


def format_retrieved_documents(documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return "No retrieved documents available for this query."

    sections: List[str] = []
    for doc in documents:
        doc_id = doc.get("doc_id", "DOC-NA")
        title = doc.get("title", "Untitled")
        source = doc.get("document_source", "Unknown source")
        passage = doc.get("passage", "").strip()
        relevance = doc.get("relevance_explanation", "").strip()
        snippet = (
            f"[{doc_id}] {title}\n"
            f"Source: {source}\n"
            f"Passage: {passage}\n"
            f"Relevance: {relevance}"
        )
        sections.append(snippet.strip())
    return "\n\n".join(sections)


def build_rag_user_prompt(
    query_record: Dict[str, Any],
    query_text: str,
    retrieved_documents: List[Dict[str, Any]],
) -> str:
    documents_block = format_retrieved_documents(retrieved_documents)
    return RAG_USER_PROMPT_TEMPLATE.format(
        company=query_record.get("company", ""),
        query_id=query_record.get("id", ""),
        query_type=query_record.get("query_type", ""),
        policy=query_record.get("policy", ""),
        category=query_record.get("category", ""),
        query_text=query_text,
        retrieved_documents_block=documents_block,
    )


def build_messages(
    system_prompt: str,
    user_prompt: str,
    payloads: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if payloads:
        examples = payloads.get("simulated_examples", {})
        for category in PAYLOAD_CATEGORY_ORDER:
            category_examples = examples.get(category, [])
            for entry in category_examples:
                messages.append({"role": "user", "content": entry["user_query"]})
                messages.append(
                    {"role": "assistant", "content": entry["assistant_response"]}
                )

    messages.append({"role": "user", "content": user_prompt})
    return messages


def process_single_query(
    args: Tuple[
        Dict,
        str,
        str,
        str,
        float,
        int,
        str,
        bool,
        Optional[Dict[str, Any]],
    ],
    lock: Manager().Lock = None
) -> None:
    """Process a single query and save result.
    
    Args:
        args: Tuple containing (query, system_prompt, model, output_path, temperature, max_tokens, field_name, verbose)
        lock: Multiprocessing lock for thread-safe file writing
    """
    (
        q,
        system_prompt,
        model,
        output_path,
        temperature,
        max_tokens,
        field_name,
        verbose,
        payloads,
    ) = args
    query_text = q[field_name]
    query_type = q.get("query_type")
    company = q.get("company")
    retrieved_documents = load_retrieved_documents(
        company=company,
        query_type=query_type,
        query_id=q.get("id"),
    )
    if not retrieved_documents:
        tqdm.write(
            f"⚠️ Retrieved documents missing for {q.get('id')} ({company}/{query_type}); continuing without grounding."
        )
    rag_user_prompt = build_rag_user_prompt(q, query_text, retrieved_documents)
    messages = build_messages(system_prompt, rag_user_prompt, payloads)
    
    try:
        response_text = call_openrouter_chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        response_text = f"__ERROR__: {str(e)}"
    
    result_obj = {
        "id": q.get("id"),
        "company": q.get("company", ""),
        "query_type": q.get("query_type", ""),
        "model": model,
        "query": query_text,
        "response": response_text,
        "retrieved_document_ids": [
            doc.get("doc_id") for doc in retrieved_documents if isinstance(doc, dict)
        ],
    }
    
    # Thread-safe write to file
    if lock:
        with lock:
            append_jsonl(result_obj, output_path)
    else:
        append_jsonl(result_obj, output_path)
    
    if verbose:
        tqdm.write(f"Query: {query_text}\nResponse: {response_text}\n---")
        tqdm.write(f"[OK] {q.get('id')}: response length={len(response_text)}")


def evaluate(
    model: str,
    company: str,
    query_type: str,
    temperature: float,
    max_tokens: int,
    verbose: bool,
    n_proc: int,
    payload_injection: bool,
    robust_system_prompts: bool,
):
    """Perform evaluation with the given model, company, and query type, then save results."""

    system_prompt, _ = load_system_prompt(company)
    queries = load_queries(company, query_type)

    payloads: Optional[Dict[str, Any]] = None
    if payload_injection:
        payloads = load_payload_examples(company)

    if payload_injection:
        output_dir = PAYLOAD_RESULT_DIR
    elif robust_system_prompts:
        output_dir = ROBUST_RESULT_DIR
    else:
        output_dir = RESULT_DIR

    output_path = os.path.join(
        output_dir,
        f"{query_type}_{company}_{model.replace('/', '_')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.jsonl",
    )

    field_name = QUERY_FIELD_MAP[query_type]

    print(
        f"Running evaluation → model: {model}, company: {company}, queries: {len(queries)}, n_proc: {n_proc}, output: {output_path}"
    )
    
    # Prepare queries with metadata
    for q in queries:
        q["company"] = company
        q["query_type"] = query_type
    
    # Create output file (empty) to ensure it exists
    with open(output_path, "w") as f:
        pass
    
    if n_proc == 1:
        # Sequential processing
        for idx, q in enumerate(tqdm(queries, desc="Evaluating queries", unit="query")):
            args = (
                q,
                system_prompt,
                model,
                output_path,
                temperature,
                max_tokens,
                field_name,
                verbose,
                payloads,
            )
            process_single_query(args)
    else:
        # Parallel processing
        manager = Manager()
        lock = manager.Lock()
        
        # Create args for each query
        args_list = [
            (
                q,
                system_prompt,
                model,
                output_path,
                temperature,
                max_tokens,
                field_name,
                verbose,
                payloads,
            )
            for q in queries
        ]
        
        # Process with multiprocessing Pool
        with Pool(processes=n_proc) as pool:
            process_func = partial(process_single_query, lock=lock)
            list(tqdm(
                pool.imap_unordered(process_func, args_list),
                total=len(queries),
                desc="Evaluating queries (parallel)",
                unit="query"
            ))

    print(f"Evaluation completed. Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models using OpenRouter API.",
    )
    parser.add_argument("--system_prompt_dir", default=SYSTEM_PROMPT_DIR, help="Directory containing system prompts")
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument(
        "--company",
        action="append",
        help="Company name; repeat or provide comma-separated values (defaults to all available)",
    )
    parser.add_argument(
        "--query_type",
        action="append",
        help="Select query set to evaluate: allowed_edge or denied_edge",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens for response",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each query and response")
    parser.add_argument("--n_proc", type=int, default=DEFAULT_N_PROC, help="Number of parallel processes for API calls")
    parser.add_argument(
        "--payload_injection",
        action="store_true",
        help="Inject company payload examples before the user query when calling the API.",
    )

    args = parser.parse_args()

    query_types_input = normalize_list_arg(args.query_type)
    if query_types_input:
        query_types = list(dict.fromkeys(query_types_input))
    else:
        query_types = list(QUERY_FIELD_MAP.keys())

    companies_input = normalize_list_arg(args.company)
    if companies_input:
        companies = list(dict.fromkeys(companies_input))
    else:
        companies = discover_companies(query_types)

    if not companies:
        raise RuntimeError("No companies found. Provide --company or ensure query files exist.")

    for company in companies:
        for query_type in query_types:
            evaluate(
                model=args.model,
                company=company,
                query_type=query_type,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
                n_proc=args.n_proc,
                payload_injection=args.payload_injection,
                robust_system_prompts=True if args.system_prompt_dir.split('/')[-1] == 'robust_system_prompts' else False,
            )
