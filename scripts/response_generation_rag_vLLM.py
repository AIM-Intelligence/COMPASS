"""Local batch response generation using vLLM."""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils.json_utils import append_jsonl

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SYSTEM_PROMPT_DIR = os.path.join(BASE_DIR, "scenario", "system_prompts")
QUERY_DIR = os.path.join(BASE_DIR, "scenario", "queries")
RESULT_DIR = os.path.join(BASE_DIR, "results", "rag_response_results")
PAYLOAD_DIR = os.path.join(BASE_DIR, "scenario", "payloads")
PAYLOAD_RESULT_DIR = os.path.join(BASE_DIR, "results", "payloads_rag_response_results")
ROBUST_RESULT_DIR = os.path.join(BASE_DIR, "results", "robust_rag_response_results")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PAYLOAD_RESULT_DIR, exist_ok=True)
os.makedirs(ROBUST_RESULT_DIR, exist_ok=True)

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

PAYLOAD_CATEGORY_ORDER = (
    "simple_refusals",
    "complex_refusals",
    "simple_acceptances",
    "complex_acceptances",
)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_BATCH_SIZE = 4


def get_query_file_path(company: str, query_type: str) -> str:
    return os.path.join(QUERY_DIR, f"verified_{query_type}", f"{company}.jsonl")


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


def normalize_list_arg(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    normalized: List[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        normalized.extend(part for part in parts if part)
    return normalized or None

def trim_system_prompt(prompt_text: str) -> str:
    delimiter = "# Context"
    if delimiter in prompt_text:
        prompt_text = prompt_text.split(delimiter)[0]
    return prompt_text.strip()


def load_system_prompt(company: str) -> tuple[str, str]:
    txt_path = os.path.join(SYSTEM_PROMPT_DIR, f"{company}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as file:
            raw_prompt = file.read().strip()
            return trim_system_prompt(raw_prompt), txt_path

    raise FileNotFoundError(
        f"System prompt for company '{company}' not found in {SYSTEM_PROMPT_DIR}."
    )


def load_queries(company: str, query_type: str) -> List[Dict[str, Any]]:
    query_file = get_query_file_path(company, query_type)
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file '{query_file}' does not exist.")

    queries: List[Dict[str, Any]] = []
    with open(query_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip() or line.lstrip().startswith("//"):
                continue
            queries.append(json.loads(line))
    return queries


def load_payload_examples(company: str) -> Dict[str, Any]:
    payload_path = os.path.join(PAYLOAD_DIR, f"{company}.json")
    if not os.path.exists(payload_path):
        raise FileNotFoundError(
            f"Payload file for company '{company}' not found at {payload_path}."
        )

    with open(payload_path, "r", encoding="utf-8") as file:
        payloads = json.load(file)

    expected_company = payloads.get("company_name")
    if expected_company and expected_company != company:
        raise ValueError(
            "Payload company name '%s' does not match requested company '%s'."
            % (expected_company, company)
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
        with open(doc_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
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
            for entry in examples.get(category, []):
                messages.append({"role": "user", "content": entry["user_query"]})
                messages.append(
                    {"role": "assistant", "content": entry["assistant_response"]}
                )

    messages.append({"role": "user", "content": user_prompt})
    return messages


def fallback_chat_prompt(messages: List[Dict[str, str]]) -> str:
    formatted: List[str] = []
    for message in messages:
        role = message.get("role", "user").capitalize()
        formatted.append(f"{role}: {message.get('content', '')}")
    formatted.append("Assistant:")
    return "\n\n".join(formatted)


def batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def messages_to_prompt(
    messages: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = True,
) -> str:
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    return fallback_chat_prompt(messages)


def derive_model_alias(model_path: str, override: Optional[str]) -> str:
    if override:
        return override
    sanitized = model_path.rstrip("/")
    return sanitized.replace("/", "_") if sanitized else model_path


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int,
    dtype: Optional[str],
    trust_remote_code: bool,
    download_dir: Optional[str],
    max_model_len: Optional[int],
) -> tuple[LLM, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if dtype:
        llm_kwargs["dtype"] = dtype
    if download_dir:
        llm_kwargs["download_dir"] = download_dir
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    print(
        "Loading model → path: %s, tensor_parallel: %d"
        % (model_path, tensor_parallel_size)
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def build_sampling_params(
    temperature: float,
    max_tokens: int,
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    seed: Optional[int],
    stop: Optional[List[str]],
) -> SamplingParams:
    sampling_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "seed": seed,
    }
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        sampling_kwargs["repetition_penalty"] = repetition_penalty

    return SamplingParams(**sampling_kwargs)


def generate_for_company(
    llm: LLM,
    tokenizer,
    alias: str,
    sampling_params: SamplingParams,
    company: str,
    query_type: str,
    batch_size: int,
    verbose: bool,
    payload_injection: bool,
    robust_system_prompts: bool,
) -> None:
    query_file = get_query_file_path(company, query_type)
    if not os.path.exists(query_file):
        print(
            "Skipping %s/%s → missing query file at %s"
            % (company, query_type, query_file)
        )
        return

    system_prompt, _ = load_system_prompt(company)
    queries = load_queries(company, query_type)
    if not queries:
        print(f"Skipping {company}/{query_type} → no queries found")
        return

    payloads: Optional[Dict[str, Any]] = None
    if payload_injection:
        payloads = load_payload_examples(company)

    if payload_injection:
        output_dir = PAYLOAD_RESULT_DIR
    elif robust_system_prompts:
        output_dir = ROBUST_RESULT_DIR
    else:
        output_dir = RESULT_DIR

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(
        output_dir,
        f"{query_type}_{company}_{alias}_{timestamp}.jsonl",
    )

    field_name = QUERY_FIELD_MAP[query_type]

    for query in queries:
        query["company"] = company
        query["query_type"] = query_type

    with open(output_path, "w", encoding="utf-8"):
        pass

    generation_inputs: List[Dict[str, Any]] = []
    for query in queries:
        query_text = query[field_name]
        retrieved_documents = load_retrieved_documents(
            company=query.get("company"),
            query_type=query.get("query_type"),
            query_id=query.get("id"),
        )
        if not retrieved_documents:
            tqdm.write(
                f"⚠️ Retrieved documents missing for {query.get('id')} ({query.get('company')}/{query.get('query_type')}); continuing without grounding."
            )
        rag_user_prompt = build_rag_user_prompt(query, query_text, retrieved_documents)
        messages = build_messages(system_prompt, rag_user_prompt, payloads)
        prompt = messages_to_prompt(messages, tokenizer)
        generation_inputs.append(
            {
                "query": query,
                "prompt": prompt,
                "messages": messages,
                "retrieved_documents": retrieved_documents,
            }
        )

    print(
        "Running evaluation → model: %s, company: %s, queries: %d, output: %s"
        % (alias, company, len(generation_inputs), output_path)
    )

    use_chat = hasattr(llm, "chat")
    progress = (
        tqdm(total=len(generation_inputs), desc="Generating", unit="query")
        if generation_inputs
        else None
    )

    for batch in batched(generation_inputs, batch_size):
        outputs = None
        if use_chat:
            try:
                chat_messages = [item["messages"] for item in batch]
                outputs = llm.chat(
                    chat_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            except Exception as exc:
                use_chat = False
                tqdm.write(
                    f"⚠️ vLLM chat generation failed for {company}/{query_type}: {exc}. Falling back to prompt generation."
                )

        if outputs is None:
            prompts = [item["prompt"] for item in batch]
            outputs = llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        for item, output in zip(batch, outputs):
            response_text = ""
            if output.outputs:
                response_text = output.outputs[0].text
            result_obj = {
                "id": item["query"].get("id"),
                "company": item["query"].get("company", ""),
                "query_type": item["query"].get("query_type", ""),
                "model": alias,
                "query": item["query"][field_name],
                "response": response_text,
                "retrieved_document_ids": [
                    doc.get("doc_id")
                    for doc in item.get("retrieved_documents", [])
                    if isinstance(doc, dict)
                ],
            }
            append_jsonl(result_obj, output_path)
            if verbose:
                tqdm.write(
                    f"Query: {result_obj['query']}\nResponse: {response_text}\n---"
                )
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    print(f"Evaluation completed. Results saved to: {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses locally using vLLM without the OpenAI API.",
    )
    parser.add_argument("--system_prompt_dir", default=SYSTEM_PROMPT_DIR, help="Directory containing system prompts")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face identifier")
    parser.add_argument(
        "--company",
        action="append",
        help="Company name; repeat or provide comma-separated values (defaults to all available)",
    )
    parser.add_argument(
        "--query_type",
        action="append",
        help="Query set to evaluate; repeat for multiples (defaults to all available)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling value (defaults to vLLM setting)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling value (defaults to vLLM setting)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty applied during sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of generation tokens",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts per generation batch",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of tensor parallel partitions",
    )
    parser.add_argument(
        "--dtype",
        help="Computation dtype (e.g. float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow executing custom model code from the repository",
    )
    parser.add_argument(
        "--download_dir",
        help="Directory to store model weights (defaults to HF cache)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        help="Override maximum model context length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--stop",
        action="append",
        help="Stop sequence; repeat flag for multiple sequences",
    )
    parser.add_argument(
        "--model_alias",
        help="Name to record in outputs (defaults to sanitized model path)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each prompt/response pair to stdout",
    )
    parser.add_argument(
        "--payload_injection",
        action="store_true",
        help="Inject company payload examples before the user query",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    alias = derive_model_alias(args.model, args.model_alias)
    llm, tokenizer = load_vllm_model(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
        max_model_len=args.max_model_len,
    )

    sampling_params = build_sampling_params(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stop=args.stop,
    )

    for company in companies:
        for query_type in query_types:
            generate_for_company(
                llm=llm,
                tokenizer=tokenizer,
                alias=alias,
                sampling_params=sampling_params,
                company=company,
                query_type=query_type,
                batch_size=args.batch_size,
                verbose=args.verbose,
                payload_injection=args.payload_injection,
                robust_system_prompts=True if args.system_prompt_dir.split('/')[-1] == 'robust_system_prompts' else False,
            )
