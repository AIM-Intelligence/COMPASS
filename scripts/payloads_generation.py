import argparse
import os
import time
from typing import Dict, List, Tuple, Optional

import yaml

from utils.json_utils import read_json, write_json
from utils.string_utils import json_style_str_to_dict
from utils.vertex_api_utils import create_response_chat
from response_judge import ResponseJudge

TARGET_PER_CATEGORY = 2

# Mapping of company names to a short industry description used in prompts/output
COMPANY_INDUSTRY_MAP: Dict[str, str] = {
    "AutoViaMotors": "Automotive manufacturing and retail",
    "CityGov": "Municipal government services",
    "FinSecure": "Banking and financial services",
    "MediCarePlus": "Healthcare provider and wellness services",
    "PlanMyTrip": "Online travel planning and booking",
    "TelePath": "Telecommunications and mobile services",
    "TutoraVerse": "Online tutoring and education",
    "VirtuRecruit": "AI-powered recruitment and career services",
}

CATEGORY_METADATA = {
    "simple_refusals": {"policy_type": "denylist"},
    "complex_refusals": {"policy_type": "denylist"},
    "simple_acceptances": {"policy_type": "allowlist"},
    "complex_acceptances": {"policy_type": "allowlist"},
}

_JUDGE: Optional[ResponseJudge] = None
JUDGE_N_PROC = 8


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_response_judge() -> ResponseJudge:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = ResponseJudge(n_proc=JUDGE_N_PROC)
    return _JUDGE


def get_company_names(scenario_dir: str) -> List[str]:
    policies_dir = os.path.join(scenario_dir, "policies")
    return [os.path.splitext(name)[0] for name in os.listdir(policies_dir) if name.endswith(".json")]


def load_company_materials(scenario_dir: str, company_name: str) -> Dict[str, str]:
    policy_path = os.path.join(scenario_dir, "policies", f"{company_name}.json")
    context_path = os.path.join(scenario_dir, "contexts", f"{company_name}.txt")

    policy = read_json(policy_path)
    with open(context_path, "r", encoding="utf-8") as handle:
        context = handle.read()

    return {"policy": policy, "context": context}


def format_policy_items(policy_dict: Dict[str, str]) -> str:
    if not policy_dict:
        return "- None provided"
    return "\n".join(f"- **{name}**: {desc}" for name, desc in policy_dict.items())


def build_prompt_messages(
    config: Dict,
    company_name: str,
    industry: str,
    materials: Dict[str, str],
) -> List[Dict[str, str]]:
    prompt_template = config["prompt_template"]
    acceptance_map = config.get("Example Acceptance Responses", {})
    refusal_map = config.get("Example Refusal Responses", {})

    acceptance_examples = acceptance_map.get(company_name)
    refusal_examples = refusal_map.get(company_name)
    if not acceptance_examples:
        raise ValueError(f"Missing acceptance examples for {company_name}")
    if not refusal_examples:
        raise ValueError(f"Missing refusal examples for {company_name}")

    prompt_body = prompt_template.replace("{acceptance_examples}", acceptance_examples.strip())
    prompt_body = prompt_body.replace("{refusal_examples}", refusal_examples.strip())

    prompt_body = prompt_body.replace(
        "[Company name, industry, and service description]",
        f"{company_name} — {industry}. Detailed context is available in the reference section below.",
    )
    prompt_body = prompt_body.replace(
        "[What the chatbot CAN discuss/provide]",
        "Refer to the Allowlist Policies in the reference materials below.",
    )
    prompt_body = prompt_body.replace(
        "[What the chatbot MUST NOT discuss/provide]",
        "Refer to the Denylist Policies in the reference materials below.",
    )

    reference_section = (
        "## Reference Materials\n\n"
        f"### Company Context\n{materials['context'].strip()}\n\n"
        "### Allowlist Policies\n"
        f"{format_policy_items(materials['policy'].get('allowlist', {})).strip()}\n\n"
        "### Denylist Policies\n"
        f"{format_policy_items(materials['policy'].get('denylist', {})).strip()}"
    )

    full_prompt = f"{prompt_body.strip()}\n\n{reference_section.strip()}"
    return [{"role": "user", "content": full_prompt}]


def validate_payload_structure(
    payload: Dict,
    company_name: str,
    expected_industry: str,
) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload is not a dict"

    if payload.get("company_name") != company_name:
        return False, "company_name does not match"

    examples = payload.get("simulated_examples")
    if not isinstance(examples, dict):
        return False, "simulated_examples missing or invalid"

    required_groups = {
        "simple_refusals": {"id", "user_query", "assistant_response", "violated_policy"},
        "complex_refusals": {"id", "user_query", "assistant_response", "violated_policies", "partial_answer_provided"},
        "simple_acceptances": {"id", "user_query", "assistant_response", "relevant_allowlist"},
        "complex_acceptances": {"id", "user_query", "assistant_response", "relevant_allowlists", "edge_case_note"},
    }

    for group_name, required_keys in required_groups.items():
        group_items = examples.get(group_name)
        if not isinstance(group_items, list) or len(group_items) != 2:
            return False, f"{group_name} must contain exactly 2 items"
        for item in group_items:
            if not isinstance(item, dict):
                return False, f"{group_name} contains non-dict entries"
            missing = required_keys - set(item.keys())
            if missing:
                return False, f"{group_name} entry missing keys: {', '.join(sorted(missing))}"
            if group_name == "complex_refusals" and not isinstance(item.get("violated_policies"), list):
                return False, "complex_refusals require violated_policies list"
            if group_name == "complex_refusals" and not isinstance(item.get("partial_answer_provided"), bool):
                return False, "complex_refusals require boolean partial_answer_provided"
            if group_name == "complex_acceptances" and not isinstance(item.get("relevant_allowlists"), list):
                return False, "complex_acceptances require relevant_allowlists list"

    return True, ""


def generate_payload_once(
    messages: List[Dict[str, str]],
    vertex_cfg: Dict,
    company_name: str,
    industry: str,
    max_api_trials: int,
) -> Dict:
    for trial in range(1, max_api_trials + 1):
        print(f"    🔄 Vertex generation trial {trial}/{max_api_trials}")
        try:
            response = create_response_chat(
                model=vertex_cfg.get("model", "gemini-2.5-pro"),
                prompt_input=messages,
                max_completion_tokens=vertex_cfg.get("max_tokens", 4096),
                temperature=vertex_cfg.get("temperature", 1.0),
                return_type="string",
            )
        except Exception as api_error:
            print(f"    ✗ API error: {api_error}")
            if trial < max_api_trials:
                time.sleep(2)
                continue
            raise

        try:
            parsed_payload = json_style_str_to_dict(response)
        except Exception as parse_error:
            print(f"    ✗ JSON parse error: {parse_error}")
            if trial < max_api_trials:
                time.sleep(1)
                continue
            raise

        is_valid, reason = validate_payload_structure(parsed_payload, company_name, industry)
        if not is_valid:
            print(f"    ✗ Validation failed: {reason}")
            if trial < max_api_trials:
                time.sleep(1)
                continue
            raise ValueError(f"Validation failed: {reason}")

        return parsed_payload

    raise RuntimeError("Failed to generate payload after retries")


def all_categories_complete(collected: Dict[str, List[Dict]]) -> bool:
    return all(len(entries) >= TARGET_PER_CATEGORY for entries in collected.values())


def evaluate_entry(company_name: str, category: str, entry: Dict) -> bool:
    judge = get_response_judge()
    policy_type = CATEGORY_METADATA[category]["policy_type"]
    item = {
        "id": f"payload-{company_name}-{policy_type}-{category}-{entry['id']}",
        "company": company_name,
        "query": entry.get("user_query", ""),
        "model": "payloads_generation",
        "query_type": category,
        "response": entry.get("assistant_response", ""),
    }

    result = judge.evaluate_response(item)
    return bool(result and result.get("is_correct"))


def load_existing_valid_examples(output_path: str, company_name: str) -> Dict[str, List[Dict]]:
    collected = {category: [] for category in CATEGORY_METADATA}

    if not os.path.exists(output_path):
        return collected

    try:
        existing_payload = read_json(output_path)
    except Exception as exc:
        print(f"    ⚠️ Failed to read existing payload {output_path}: {exc}")
        return collected

    examples = existing_payload.get("simulated_examples", {})
    for category, entries in examples.items():
        if category not in collected:
            continue
        for entry in entries:
            if len(collected[category]) >= TARGET_PER_CATEGORY:
                break
            if evaluate_entry(company_name, category, entry):
                collected[category].append(entry)
            else:
                print(f"    ⚠️ Existing example for {category} id={entry.get('id')} failed validation and will be replaced")

    return collected


def build_final_payload(company_name: str, industry: str, collected: Dict[str, List[Dict]]) -> Dict:
    payload = {
        "company_name": company_name,
        "industry": industry,
        "simulated_examples": {},
    }

    for category, entries in collected.items():
        normalized = []
        for idx, entry in enumerate(entries, start=1):
            normalized_entry = dict(entry)
            normalized_entry["id"] = idx
            normalized.append(normalized_entry)
        payload["simulated_examples"][category] = normalized

    return payload


def process_company(
    config: Dict,
    scenario_dir: str,
    company_name: str,
    output_dir: str,
    overwrite: bool = False,
) -> bool:
    industry = COMPANY_INDUSTRY_MAP.get(company_name)
    if not industry:
        raise ValueError(f"Industry mapping not found for {company_name}")

    output_path = os.path.join(output_dir, f"{company_name}.json")
    if os.path.exists(output_path) and not overwrite:
        print(f"    ✓ {company_name}.json already exists, skipping")
        return True

    materials = load_company_materials(scenario_dir, company_name)
    messages = build_prompt_messages(config, company_name, industry, materials)

    vertex_cfg = config.get("vertex", {})
    max_trials = config.get("retry", {}).get("max_trials", 3)

    collected: Dict[str, List[Dict]]
    if os.path.exists(output_path):
        collected = load_existing_valid_examples(output_path, company_name)
        for category, entries in collected.items():
            if entries:
                print(f"    • Reusing {len(entries)} validated entries for {category}")
    else:
        collected = {category: [] for category in CATEGORY_METADATA}

    if all_categories_complete(collected):
        print("    ✓ Existing payload already satisfies validation requirements")
        write_json(build_final_payload(company_name, industry, collected), output_path)
        return True

    generation_round = 0
    while not all_categories_complete(collected) and generation_round < max_trials:
        generation_round += 1
        print(f"    ▶️ Generation round {generation_round}/{max_trials}")
        try:
            parsed_payload = generate_payload_once(messages, vertex_cfg, company_name, industry, max_trials)
        except Exception as error:
            print(f"    ✗ Generation failed: {error}")
            if generation_round < max_trials:
                time.sleep(2)
                continue
            raise

        examples = parsed_payload.get("simulated_examples", {})
        for category, entries in examples.items():
            if category not in collected:
                continue
            for entry in entries:
                if len(collected[category]) >= TARGET_PER_CATEGORY:
                    continue
                if evaluate_entry(company_name, category, entry):
                    collected[category].append(entry)
                else:
                    print(f"    ⚠️ Rejected {category} id={entry.get('id')} due to failed policy check")

    if not all_categories_complete(collected):
        missing = {category: TARGET_PER_CATEGORY - len(entries) for category, entries in collected.items() if len(entries) < TARGET_PER_CATEGORY}
        raise RuntimeError(f"Unable to gather enough valid examples for {company_name}: {missing}")

    write_json(build_final_payload(company_name, industry, collected), output_path)
    print(f"    ✓ Saved payload to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate chatbot payload examples per company")
    parser.add_argument("--company", action="append", help="Specific company to process (can be repeated)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode and limit processed companies")
    parser.add_argument("--max-companies", type=int, help="Maximum number of companies to process in debug")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing payload files")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scenario_dir = os.path.join(project_root, "scenario")
    output_dir = os.path.join(scenario_dir, "payloads")
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(script_dir, "config", "payloads_generation.yaml")
    print("Loading configuration...")
    config = load_config(config_path)

    debug_enabled = args.debug or config.get("debug", {}).get("enabled", False)
    max_companies = args.max_companies or config.get("debug", {}).get("max_companies", 3)

    company_names = get_company_names(scenario_dir)
    company_names = sorted(company_names)

    if args.company:
        requested = [name.strip() for name in args.company]
        invalid = [name for name in requested if name not in company_names]
        if invalid:
            available = ", ".join(company_names)
            raise ValueError(f"Unknown company names requested: {', '.join(invalid)}. Available: {available}")
        company_names = requested

    if debug_enabled:
        company_names = company_names[:max_companies]
        print(f"🐛 Debug mode active: limiting to {len(company_names)} companies")

    print(f"Processing companies: {', '.join(company_names)}")

    successes = 0
    for idx, company_name in enumerate(company_names, start=1):
        print(f"\n[{idx}/{len(company_names)}] {company_name}")
        if process_company(config, scenario_dir, company_name, output_dir, overwrite=args.overwrite):
            successes += 1

    print(f"\nCompleted with {successes}/{len(company_names)} successful payloads")


if __name__ == "__main__":
    main()
