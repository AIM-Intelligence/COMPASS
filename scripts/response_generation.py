"""
Unified Response Generation Script

Generates model responses using any supported API provider (OpenAI, Anthropic, Vertex, OpenRouter).
The provider is automatically selected based on the config file or CLI arguments.

Usage:
    python response_generation.py --model "claude-sonnet-4-20250514" --company "MyOrg" --query_type "base"
    python response_generation.py --config config/response_generation.yaml --company "MyOrg" --query_type "base"
"""

import argparse
import json
import os
import time
import yaml
from datetime import datetime
from functools import partial
from multiprocessing import Pool, Manager
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from tqdm import tqdm

from utils.json_utils import append_jsonl
from utils.unified_api_utils import (
    create_response_chat,
    get_provider_from_config,
    check_api_key,
    get_required_env_var,
    SUPPORTED_PROVIDERS,
)

# Load environment variables
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SYSTEM_PROMPT_DIR = os.path.join(BASE_DIR, "scenario", "system_prompts")
QUERY_DIR = os.path.join(BASE_DIR, "scenario", "queries")
RESULT_DIR = os.path.join(BASE_DIR, "results", "response_results")
PAYLOAD_DIR = os.path.join(BASE_DIR, "scenario", "payloads")
PAYLOAD_RESULT_DIR = os.path.join(BASE_DIR, "results", "payloads_response_results")
ROBUST_RESULT_DIR = os.path.join(BASE_DIR, "results", "robust_response_results")

# Ensure results directories exist
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PAYLOAD_RESULT_DIR, exist_ok=True)
os.makedirs(ROBUST_RESULT_DIR, exist_ok=True)

# Map query_type to field names inside jsonl
QUERY_FIELD_MAP = {
    "base": "base_query",
    "allowed_edge": "allowed_edge_query",
    "denied_edge": "denied_edge_query",
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_RETRY_COUNT = 30
DEFAULT_RETRY_DELAY = 2  # seconds
DEFAULT_N_PROC = 1

PAYLOAD_CATEGORY_ORDER = (
    "simple_refusals",
    "complex_refusals",
    "simple_acceptances",
    "complex_acceptances",
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_system_prompt(company: str, system_prompt_dir: str = SYSTEM_PROMPT_DIR) -> Tuple[str, str]:
    """Load the system prompt text for the given company, returning (text, path)."""
    txt_path = os.path.join(system_prompt_dir, f"{company}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip(), txt_path

    raise FileNotFoundError(
        f"System prompt for company '{company}' not found in {system_prompt_dir}."
    )


def load_queries(company: str, query_type: str) -> List[Dict]:
    """Load query objects for a company and type (base / allowed_edge / denied_edge)."""
    query_file = os.path.join(QUERY_DIR, f"verified_{query_type}", f"{company}.jsonl")
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")

    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            data = json.loads(line)
            queries.append(data)
    return queries


def load_payload_examples(company: str) -> Dict[str, Any]:
    """Load payload examples for few-shot injection."""
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


def build_messages(
    system_prompt: str,
    query_text: str,
    payloads: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Build messages list for chat completion."""
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

    messages.append({"role": "user", "content": query_text})
    return messages


def call_api_with_retry(
    config: dict,
    messages: List[Dict[str, str]],
    retries: int = DEFAULT_RETRY_COUNT,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """Call API with retry logic using unified API utils.
    
    Args:
        config: Configuration dictionary with API settings
        messages: List of message dictionaries
        retries: Number of retry attempts
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        Response text from the API
        
    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None
    current_delay = retry_delay
    
    for attempt in range(1, retries + 1):
        try:
            response_text = create_response_chat(
                config=config,
                prompt_input=messages,
                return_type="string",
            )
            return response_text
        except Exception as e:
            last_exception = e
            if attempt < retries:
                tqdm.write(f"⚠️ API call failed (attempt {attempt}/{retries}): {str(e)}")
                tqdm.write(f"Retrying in {current_delay:.1f} seconds...")
                time.sleep(current_delay)
                current_delay *= 1.5  # Exponential backoff
            else:
                tqdm.write(f"❌ All {retries} attempts failed for API call")
    
    raise last_exception


def process_single_query(
    args: Tuple[
        Dict,  # query
        str,   # system_prompt
        dict,  # config
        str,   # output_path
        str,   # field_name
        bool,  # verbose
        Optional[Dict[str, Any]],  # payloads
    ],
    lock=None
) -> None:
    """Process a single query and save result."""
    (
        q,
        system_prompt,
        config,
        output_path,
        field_name,
        verbose,
        payloads,
    ) = args
    
    query_text = q[field_name]
    messages = build_messages(system_prompt, query_text, payloads)

    try:
        response_text = call_api_with_retry(config=config, messages=messages)
    except Exception as e:
        response_text = f"__ERROR__: {str(e)}"

    # Get model name from config
    if 'api' in config and 'model' in config['api']:
        model = config['api']['model']
    else:
        # Legacy format - find provider key
        for provider in SUPPORTED_PROVIDERS:
            if provider in config:
                model = config[provider].get('model', 'unknown')
                break
        else:
            model = 'unknown'

    result_obj = {
        "id": q.get("id"),
        "company": q.get("company", ""),
        "query_type": q.get("query_type", ""),
        "model": model,
        "query": query_text,
        "response": response_text,
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
    config: dict,
    company: str,
    query_type: str,
    verbose: bool,
    n_proc: int,
    payload_injection: bool,
    robust_system_prompts: bool,
    system_prompt_dir: str = SYSTEM_PROMPT_DIR,
) -> None:
    """Run evaluation and save results."""
    system_prompt, _ = load_system_prompt(company, system_prompt_dir)
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

    # Get model name for filename
    if 'api' in config and 'model' in config['api']:
        model = config['api']['model']
    else:
        for provider in SUPPORTED_PROVIDERS:
            if provider in config:
                model = config[provider].get('model', 'unknown')
                break
        else:
            model = 'unknown'

    output_path = os.path.join(
        output_dir,
        f"{query_type}_{company}_{model.replace('/', '_')}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.jsonl",
    )

    field_name = QUERY_FIELD_MAP[query_type]

    # Get provider info for display
    provider = get_provider_from_config(config)
    
    print(f"📡 Using API provider: {provider}")
    print(
        f"Running evaluation → model: {model}, company: {company}, "
        f"queries: {len(queries)}, n_proc: {n_proc}, output: {output_path}"
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
        for q in tqdm(queries, desc="Evaluating queries", unit="query"):
            args = (
                q,
                system_prompt,
                config,
                output_path,
                field_name,
                verbose,
                payloads,
            )
            process_single_query(args)
    else:
        # Parallel processing
        manager = Manager()
        lock = manager.Lock()

        args_list = [
            (
                q,
                system_prompt,
                config,
                output_path,
                field_name,
                verbose,
                payloads,
            )
            for q in queries
        ]

        with Pool(processes=n_proc) as pool:
            process_func = partial(process_single_query, lock=lock)
            list(
                tqdm(
                    pool.imap_unordered(process_func, args_list),
                    total=len(queries),
                    desc="Evaluating queries (parallel)",
                    unit="query",
                )
            )

    print(f"✅ Evaluation completed. Results saved to: {output_path}")


def build_config_from_args(args) -> dict:
    """Build config dictionary from command line arguments."""
    config = {
        'api': {
            'provider': args.provider,
            'model': args.model,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
    }
    
    # Add provider-specific settings
    if args.provider == 'vertex':
        if args.region:
            config['api']['region'] = args.region
        if args.project_id:
            config['api']['project_id'] = args.project_id
    
    if args.provider == 'openai' and args.reasoning_effort:
        config['api']['reasoning_effort'] = args.reasoning_effort
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run response generation using any supported API provider.",
    )
    
    # Config file option
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file. If provided, overrides other API settings."
    )
    
    # API settings (used if no config file)
    parser.add_argument(
        "--provider",
        type=str,
        choices=SUPPORTED_PROVIDERS,
        default="openai",
        help="API provider to use (default: openai)"
    )
    parser.add_argument("--model", type=str, help="Model identifier (required if no config)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max completion tokens per response",
    )
    
    # Provider-specific settings
    parser.add_argument("--region", type=str, help="Vertex AI region (for vertex provider)")
    parser.add_argument("--project_id", type=str, help="Vertex AI project ID (for vertex provider)")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        help="Reasoning effort for OpenAI reasoning models (e.g., 'medium')"
    )
    
    # Required arguments
    parser.add_argument("--company", required=True, help="Company name matching prompt and query files")
    parser.add_argument(
        "--query_type",
        required=True,
        choices=list(QUERY_FIELD_MAP.keys()),
        help="Choose which set of queries to evaluate: base, allowed_edge, denied_edge",
    )
    
    # Optional settings
    parser.add_argument(
        "--system_prompt_dir",
        default=SYSTEM_PROMPT_DIR,
        help="Directory containing system prompts"
    )
    parser.add_argument("--verbose", action="store_true", help="Print each query and response")
    parser.add_argument(
        "--n_proc",
        type=int,
        default=DEFAULT_N_PROC,
        help="Number of parallel processes for API calls",
    )
    parser.add_argument(
        "--payload_injection",
        action="store_true",
        help="Inject company payload examples before the user query.",
    )

    args = parser.parse_args()

    # Load or build config
    if args.config:
        config = load_config(args.config)
        print(f"📋 Loading configuration from: {args.config}")
    else:
        if not args.model:
            parser.error("--model is required when not using --config")
        config = build_config_from_args(args)
    
    # Check API key
    provider = get_provider_from_config(config)
    if not check_api_key(config):
        print(f"❌ ERROR: {get_required_env_var(provider)} environment variable not set")
        exit(1)

    # Determine if using robust system prompts
    robust_system_prompts = args.system_prompt_dir.split('/')[-1] == 'robust_system_prompts'

    evaluate(
        config=config,
        company=args.company,
        query_type=args.query_type,
        verbose=args.verbose,
        n_proc=args.n_proc,
        payload_injection=args.payload_injection,
        robust_system_prompts=robust_system_prompts,
        system_prompt_dir=args.system_prompt_dir,
    )
