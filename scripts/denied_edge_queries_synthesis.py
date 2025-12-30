import os
import glob
import argparse
import time
import json
import random

import yaml
import requests
import dotenv
from utils.json_utils import read_jsonl, write_jsonl, append_jsonl
from utils.string_utils import json_style_str_to_dict
from typing import List, Optional, Tuple

# Load environment variables
dotenv.load_dotenv()


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_company_names(scenario_dir: str) -> list:
    """Gets list of company names from scenario directory."""
    policies_dir = os.path.join(scenario_dir, 'policies')
    policy_files = glob.glob(os.path.join(policies_dir, '*.json'))
    company_names = [os.path.splitext(os.path.basename(f))[0] for f in policy_files]
    return company_names


def load_verified_base_queries(scenario_dir: str, company_name: str) -> list:
    """Loads denylist entries from verified_base directory for a specific company."""
    verified_base_path = os.path.join(scenario_dir, 'queries', 'verified_base', f'{company_name}.jsonl')
    
    if not os.path.exists(verified_base_path):
        raise FileNotFoundError(f"Verified base file not found: {verified_base_path}")
    
    # Read JSONL file
    all_queries = read_jsonl(verified_base_path)
    
    # Filter only denylist entries
    denylist_queries = [query for query in all_queries if query.get('category') == 'denylist']
    
    if not denylist_queries:
        raise ValueError(f"No denylist entries found in {verified_base_path}")
    
    return denylist_queries


def validate_denied_edge_response(parsed_response: dict, expected_attack_types: Optional[List[str]] = None) -> bool:
    """Validates if generated denied_edge response matches expected structure."""
    try:
        # Check required keys
        if "original_query" not in parsed_response:
            return False
        # Determine which key contains variation mapping (default attack_variations)
        variations_key = None
        if "attack_variations" in parsed_response:
            variations_key = "attack_variations"
        else:
            return False
        
        # Check attack variations structure
        attack_variations = parsed_response[variations_key]
        # If an expected list is provided, ensure response contains all keys
        if expected_attack_types:
            for attack_type in expected_attack_types:
                if attack_type not in attack_variations:
                    return False
        
        # Further validate each provided variation
        types_to_check = expected_attack_types if expected_attack_types else attack_variations.keys()
        for attack_type in types_to_check:
            if attack_type not in attack_variations:
                return False
            if not isinstance(attack_variations[attack_type], str):
                return False
            if len(attack_variations[attack_type].strip()) == 0:
                return False
        
        return True
    except Exception as e:
        print(f"    ⚠️ denied_edge JSON validation error: {str(e)}")
        return False


def create_denied_edge_prompt(config: dict, base_query: str) -> Tuple[str, List[str]]:
    """Creates prompt for denied_edge query generation. Looks for any key that starts
    with "prompt_template" if the default key is not present so that different YAML
    files (e.g., prompt_template_short/long) are supported without code changes.

    Randomly samples attack strategies (deterministically per query) based on the
    configured number_of_queries and fills the appropriate placeholders.
    """

    prompt_template = (
        config.get('prompt_template')
        or config.get('prompt_template_short')
        or config.get('prompt_template_long')
    )

    if not prompt_template:
        raise KeyError(
            "No prompt_template found in YAML (expected keys: prompt_template, "
            "prompt_template_short, prompt_template_long)"
        )

    attack_strategies = config.get('attack_strategies')
    if not isinstance(attack_strategies, dict) or not attack_strategies:
        raise ValueError("attack_strategies must be a non-empty mapping in the config")

    if 'number_of_queries' not in config:
        raise KeyError("number_of_queries not found in the configuration")

    try:
        number_of_queries = int(config['number_of_queries'])
    except (TypeError, ValueError) as exc:
        raise ValueError("number_of_queries must be an integer") from exc

    strategy_names = list(attack_strategies.keys())
    if number_of_queries > len(strategy_names):
        raise ValueError(
            f"number_of_queries ({number_of_queries}) exceeds available attack_strategies ({len(strategy_names)})"
        )

    # Deterministic sampling per base_query so results can be reproduced.
    rng = random.Random(len(base_query))
    selected_strategies = rng.sample(strategy_names, k=number_of_queries)

    ordinal_names = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    ]

    if number_of_queries > len(ordinal_names):
        raise ValueError("number_of_queries exceeds supported ordinal placeholders in the prompt template")

    replacements = {
        'base_query': base_query,
        'original_query': base_query,
        'number_of_queries': str(number_of_queries),
    }

    for idx, strategy_name in enumerate(selected_strategies):
        ordinal = ordinal_names[idx]
        replacements[f'{ordinal}_strategy'] = strategy_name
        strategy_guide = attack_strategies.get(strategy_name, "")
        replacements[f'{ordinal}_strategy_guide'] = strategy_guide.strip()

    user_prompt = prompt_template
    for placeholder, value in replacements.items():
        user_prompt = user_prompt.replace(f'{{{placeholder}}}', value)

    return user_prompt, selected_strategies


def call_openrouter_api(config: dict, prompt: str, max_retries: int = 3) -> str:
    """Makes API call to OpenRouter."""
    url = config['openrouter']['url']
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv('YOUR_SITE_URL', 'https://localhost'),
        "X-Title": os.getenv('YOUR_SITE_NAME', 'denied_edge Query Synthesis2')
    }
    
    data = {
        "model": config['openrouter']['model'],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": config['openrouter']['temperature'],
        "max_tokens": config['openrouter']['max_tokens'],
        "stream": config['openrouter']['stream']
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(data),
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return response_data['choices'][0]['message']['content']
                else:
                    raise Exception(f"No choices in response: {response_data}")
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"    ⚠️ API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e


def process_single_query(config: dict, query_item: dict) -> dict:
    """Processes a single query item to generate denied_edge variations."""
    base_query = query_item['base_query']
    
    # Create prompt for this specific query
    prompt, expected_attack_types = create_denied_edge_prompt(config, base_query)
    
    # Get retry settings
    max_trials = config.get('retry', {}).get('max_trials', 3)
    
    for trial in range(1, max_trials + 1):
        try:
            # Call OpenRouter API
            response = call_openrouter_api(config, prompt)
            
            # Parse JSON response
            try:
                parsed_response = json_style_str_to_dict(response)
                print(f"      ✓ Successfully parsed denied_edge response for query {query_item['id']}")
            except Exception as parse_error:
                print(f"      ⚠️ Failed to parse JSON from response: {str(parse_error)}")
                if trial < max_trials:
                    print(f"      🔄 Retrying due to JSON parsing error...")
                    time.sleep(1)
                    continue
                else:
                    # Return with error if parsing fails
                    result = query_item.copy()
                    result['id'] = f"denied_edge-{query_item['id']}-parse_error"
                    result['attack_variation'] = "parse_error"
                    result['denied_edge_query'] = f"Error: {str(parse_error)}"
                    result['raw_response'] = response
                    return [result]
            
            # Validate response structure
            if isinstance(parsed_response, dict) and "raw_response" not in parsed_response:
                expected_list = expected_attack_types or (
                    config.get('expected_attack_types') if isinstance(config.get('expected_attack_types'), list) else None
                )
                is_valid = validate_denied_edge_response(parsed_response, expected_list)
                if not is_valid:
                    print(f"      ⚠️ denied_edge response validation failed")
                    if trial < max_trials:
                        print(f"      🔄 Retrying due to invalid response structure...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"      ✗ Final attempt failed - saving invalid structure")
            
            # Detect variations key dynamically
            variations_key = "attack_variations"
            for k in parsed_response.keys():
                if k.endswith("_variations"):
                    variations_key = k
                    break
            attack_variations = parsed_response.get(variations_key, {})
            
            # Create multiple results - one for each attack variation
            results = []
            base_id = f"denied_edge-{query_item['id']}"
            
            if attack_variations:
                for attack_type, denied_edge_query in attack_variations.items():
                    if denied_edge_query and denied_edge_query.strip():
                        result = query_item.copy()
                        result['id'] = f"{base_id}-{attack_type}"
                        result['attack_variation'] = attack_type
                        result['denied_edge_query'] = denied_edge_query
                        results.append(result)
            
            # If no valid attack variations found, create a single error result
            if not results:
                result = query_item.copy()
                result['id'] = f"{base_id}-error"
                result['attack_variation'] = "error"
                result['denied_edge_query'] = "No variations generated"
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"      ✗ Error on trial {trial}: {str(e)}")
            if trial < max_trials:
                print(f"      🔄 Retrying...")
                time.sleep(2)
                continue
            else:
                print(f"      ✗ All trials failed for query {query_item['id']}")
                result = query_item.copy()
                result['id'] = f"denied_edge-{query_item['id']}-api_error"
                result['attack_variation'] = "api_error"
                result['denied_edge_query'] = f"Error: {str(e)}"
                return [result]
    
    return None


def process_company_with_multiple_configs(config_paths: list, scenario_dir: str, company_name: str, output_dir: str, debug_limit: int = None, overwrite: bool = False):
    """Processes all denylist queries for a specific company using multiple config files."""
    
    # Build dated output filename: e.g., AutoViaMotors.jsonl
    output_filename = f"{company_name}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    # If the exact dated file already exists, skip to avoid re-processing same day
    if os.path.exists(output_path):
        if overwrite:
            print(f"    🔄 Overwrite enabled - regenerating {output_filename}")
            os.remove(output_path)
        else:
            print(f"    ✓ {output_filename} already exists, skipping processing")
            return True
    
    try:
        # Load denylist queries from verified_base
        denylist_queries = load_verified_base_queries(scenario_dir, company_name)
        print(f"    📋 Loaded {len(denylist_queries)} denylist queries")
        
        all_denied_edge_results = []
        total_processed = 0

        # Create/prepare output file early so we can append to it incrementally
        open(output_path, "a", encoding="utf-8").close()

        # Keep a set of IDs already written to avoid duplicates during retries
        processed_ids = set()
        
        # Apply debug limit if specified
        queries_to_process = denylist_queries
        if debug_limit:
            queries_to_process = denylist_queries[:debug_limit]
            print(f"    🐛 Debug mode: limiting to {len(queries_to_process)} queries")
        
        # Process with each config file
        for config_idx, config_path in enumerate(config_paths, 1):
            print(f"    📄 Processing with config {config_idx}/{len(config_paths)}: {os.path.basename(config_path)}")
            config = load_config(config_path)
            
            # Process each denylist query with current config
            for i, query_item in enumerate(queries_to_process, 1):
                print(f"      [{i}/{len(queries_to_process)}] Config {config_idx}: {query_item['base_query'][:60]}...")
                
                result_list = process_single_query(config, query_item)
                if result_list:
                    # Append each result immediately for resilience to interruptions
                    for res in result_list:
                        # Make ID unique across configs by adding config suffix
                        config_suffix = os.path.splitext(os.path.basename(config_path))[0]
                        res["id"] = f"{res['id']}-{config_suffix}"
                        res["config_source"] = config_suffix
                        
                        if res["id"] not in processed_ids:
                            append_jsonl(res, output_path)
                            processed_ids.add(res["id"])
                            all_denied_edge_results.append(res)
                    total_processed = len(processed_ids)
        
        # Finally, rewrite the file with de-duplicated, ordered results
        write_jsonl(all_denied_edge_results, output_path)
        print(f"    ✅ Saved {total_processed} denied_edge query variations for {company_name} (using {len(config_paths)} configs)")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {company_name}: {str(e)}")
        return False


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str, debug_limit: int = None, overwrite: bool = False):
    """Processes all denylist queries for a specific company."""
    
    # Build dated output filename: e.g., AutoViaMotors.jsonl
    output_filename = f"{company_name}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    # If the exact dated file already exists, skip to avoid re-processing same day
    if os.path.exists(output_path):
        if overwrite:
            print(f"    🔄 Overwrite enabled - regenerating {output_filename}")
            os.remove(output_path)
        else:
            print(f"    ✓ {output_filename} already exists, skipping processing")
            return True
    
    try:
        # Load denylist queries from verified_base
        denylist_queries = load_verified_base_queries(scenario_dir, company_name)
        print(f"    📋 Loaded {len(denylist_queries)} denylist queries")
        
        denied_edge_results = []
        total_processed = 0

        # Create/prepare output file early so we can append to it incrementally
        # If the file already exists we will append new items and later overwrite
        # with the final consolidated list to remove any accidental duplicates.
        open(output_path, "a", encoding="utf-8").close()

        # Keep a set of IDs already written to avoid duplicates during retries
        processed_ids = set()
        
        # Apply debug limit if specified
        queries_to_process = denylist_queries
        if debug_limit:
            queries_to_process = denylist_queries[:debug_limit]
            print(f"    🐛 Debug mode: limiting to {len(queries_to_process)} queries")
        
        # Process each denylist query
        for i, query_item in enumerate(queries_to_process, 1):
            print(f"      [{i}/{len(queries_to_process)}] Processing: {query_item['base_query'][:80]}...")
            
            result_list = process_single_query(config, query_item)
            if result_list:
                # Append each result immediately for resilience to interruptions
                for res in result_list:
                    if res["id"] not in processed_ids:
                        append_jsonl(res, output_path)
                        processed_ids.add(res["id"])
                        denied_edge_results.append(res)
                total_processed = len(processed_ids)
        
        # Finally, rewrite the file with de-duplicated, ordered results
        write_jsonl(denied_edge_results, output_path)
        print(f"    ✅ Saved {total_processed} denied_edge query variations for {company_name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {company_name}: {str(e)}")
        return False


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate denied_edge variations of denylist queries')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies and queries)')
    parser.add_argument('--max_companies', type=int, help='Maximum number of companies to process (used in debug mode)')
    parser.add_argument('--queries_per_category', type=int, help='Maximum queries per category in debug mode')
    parser.add_argument('--company', type=str, help='Specific company name to process (if not specified, processes all companies)')
    parser.add_argument('--overwrite', action='store_true', help='Reprocess and overwrite output even if dated file exists')
    parser.add_argument('--config', '-c', type=str, nargs='+', help='Path(s) to YAML configuration file(s). Can specify multiple configs to combine results.')
    parser.add_argument('--multi_config', action='store_true', help='Use both short and long config files automatically')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scenario_dir = os.path.join(project_root, 'scenario')
    output_dir = os.path.join(scenario_dir, 'queries', 'denied_edge')
    
    # Determine configuration file paths
    config_paths = []
    if args.multi_config:
        # Use both short and long configs automatically
        short_config = os.path.join(script_dir, 'config', 'denied_edge_queries_synthesis_short.yaml')
        long_config = os.path.join(script_dir, 'config', 'denied_edge_queries_synthesis_long.yaml')
        config_paths = [short_config, long_config]
        print("🔄 Multi-config mode: using both short and long configurations")
    elif args.config:
        # Use specified config file(s)
        for config_arg in args.config:
            config_path = config_arg if os.path.isabs(config_arg) else os.path.join(os.getcwd(), config_arg)
            config_paths.append(config_path)
    else:
        print("❌ ERROR: No configuration file specified")
        return
    
    # Validate config files exist
    for config_path in config_paths:
        if not os.path.exists(config_path):
            print(f"❌ ERROR: Configuration file not found: {config_path}")
            return
    
    # Check/create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration (for debug settings - use first config)
    print(f"📋 Loading configuration(s): {[os.path.basename(cp) for cp in config_paths]}")
    config = load_config(config_paths[0])  # Use first config for debug settings
    
    # Debug mode settings
    debug_enabled = args.debug or config.get('debug', {}).get('enabled', False)
    max_companies = args.max_companies or config.get('debug', {}).get('max_companies', 1)
    queries_per_category = args.queries_per_category or (10 if debug_enabled else None)
    
    if debug_enabled:
        print(f"🐛 Debug mode enabled - processing maximum {max_companies} companies")
        if queries_per_category:
            print(f"🐛 Debug mode: limiting to {queries_per_category} queries per category")
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key in the .env file or environment")
        return
    
    # Get company names list
    print("🔍 Getting company names...")
    all_company_names = get_company_names(scenario_dir)
    print(f"Found {len(all_company_names)} companies: {', '.join(all_company_names)}")
    
    # Filter by specific company if specified
    if args.company:
        if args.company in all_company_names:
            company_names = [args.company]
            print(f"🎯 Processing specific company: {args.company}")
        else:
            print(f"❌ ERROR: Company '{args.company}' not found in scenario directory")
            print(f"Available companies: {', '.join(all_company_names)}")
            return
    else:
        company_names = all_company_names
    
    # Limit number of companies in debug mode (only if no specific company specified)
    if debug_enabled and not args.company:
        company_names = company_names[:max_companies]
        print(f"🐛 Debug mode: processing {len(company_names)} companies: {', '.join(company_names)}")
    
    # Process each company
    successful = 0
    total = len(company_names)
    
    for i, company_name in enumerate(company_names, 1):
        print(f"\n[{i}/{total}] 🏢 Processing {company_name}...")
        
        # Use multiple configs if specified, otherwise use single config
        if len(config_paths) > 1:
            if process_company_with_multiple_configs(config_paths, scenario_dir, company_name, output_dir, queries_per_category, args.overwrite):
                successful += 1
        else:
            if process_company(config, scenario_dir, company_name, output_dir, queries_per_category, args.overwrite):
                successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Results saved to: {output_dir}")
    
    if debug_enabled:
        print(f"🐛 Debug mode was enabled - processed {total} out of {len(get_company_names(scenario_dir))} total companies")
        if queries_per_category:
            print(f"🐛 Limited to {queries_per_category} queries per category")


if __name__ == "__main__":
    main()
