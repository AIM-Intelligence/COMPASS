import os
import glob
import argparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import yaml
from utils.json_utils import read_json, read_jsonl, write_jsonl, append_jsonl
from utils.openai_api_utils import create_response_chat_for_structured_output_of_reasoning_model
from utils.string_utils import response2dict, json_style_str_to_dict


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


def filter_company_names(all_companies: list, selected_companies: list | None) -> list:
    """Filters company list based on user selection."""
    if not selected_companies:
        return all_companies

    normalized = [name.strip() for name in selected_companies if name and name.strip()]
    ordered_unique = []
    seen = set()
    for name in normalized:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)

    missing = [name for name in ordered_unique if name not in all_companies]
    if missing:
        raise ValueError(f"Unknown company name(s): {', '.join(missing)}")

    return [name for name in ordered_unique if name in all_companies]


def load_company_data(scenario_dir: str, company_name: str) -> dict:
    """Loads policy, context, and denied_edge queries data for a specific company."""
    # Load policy file
    policy_path = os.path.join(scenario_dir, 'policies', f'{company_name}.json')
    policy_data = read_json(policy_path)
    
    # Load context file
    context_path = os.path.join(scenario_dir, 'contexts', f'{company_name}.txt')
    with open(context_path, 'r', encoding='utf-8') as file:
        context_data = file.read()
    
    # Load denied_edge queries file
    denied_edge_queries_path = os.path.join(scenario_dir, 'queries', 'denied_edge', f'{company_name}.jsonl')
    denied_edge_queries_data = []
    
    import json
    with open(denied_edge_queries_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                denied_edge_queries_data.append(json.loads(line.strip()))
    
    return {
        'policy': policy_data,
        'context': context_data,
        'denied_edge_queries': denied_edge_queries_data
    }


def create_messages(config: dict, company_data: dict, query: str) -> list:
    """Creates message list for Anthropic API call."""
    prompt_template = config['prompt_template']
    
    # Replace placeholders with actual data
    user_prompt = prompt_template.replace('{company_context}', company_data['context'])
    
    # Convert policy data to JSON string and insert
    import json
    policy_json = json.dumps(company_data['policy'], indent=2, ensure_ascii=False)
    user_prompt = user_prompt.replace('{policy_document}', policy_json)
    user_prompt = user_prompt.replace('{query}', query)
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def call_policy_matcher_api(config: dict, company_data: dict, query: str, max_trials: int = 3, verbose: bool = False) -> dict:
    """Calls the policy matcher API for a single query."""
    messages = create_messages(config, company_data, query)
    response_schema = {
        "type": "json_schema",
        "json_schema": config['output_json_schema']
    }

    for trial in range(1, max_trials + 1):
        try:
            if verbose and trial > 1:
                print(f"          🔄 Retry attempt {trial}/{max_trials}")
            
            response = create_response_chat_for_structured_output_of_reasoning_model(
                model=config['openai']['model'],
                prompt_input=messages,
                response_schema=response_schema,
                max_completion_tokens=config['openai']['max_tokens'],
                temperature=config['openai']['temperature'],
                reasoning_effort=config['openai'].get('reasoning_effort'),
                return_type="json"
            )
            
            if verbose:
                print(f"          ✓ API call successful on trial {trial}")
            
            # Response is already parsed when using structured output
            if isinstance(response, dict):
                return response
            else:
                # If return_type is "json", response is already a dict
                return response
        except Exception as api_error:
            error_msg = f"        ✗ API error on trial {trial}: {str(api_error)}"
            if verbose:
                print(error_msg)
            elif trial == max_trials:  # Only print on final failure if not verbose
                print(error_msg)
            
            if trial < max_trials:
                if verbose:
                    print(f"          ⏳ Waiting 2 seconds before retry...")
                time.sleep(2)
                continue
            else:
                return {"error": f"API error: {str(api_error)}"}
    
    return {"error": "All trials failed"}


def apply_post_processing_logic(query_result: dict, original_policy: str, category_type: str, policy_list_type: str) -> dict:
    """
    Applies post-processing logic for out-policy verification only.
    
    Args:
        query_result: API response with matching_out_policies
        original_policy: The original policy name from base queries
        category_type: Should be 'out-policy'
        policy_list_type: Should be 'denylist'
    
    Returns:
        dict with verification result
    """
    if "matching_out_policies" not in query_result:
        return {
            "verification_result": "failed",
            "reason": "No matching_out_policies found in API response",
            "original_policy": original_policy,
            "category_type": category_type,
            "policy_list_type": policy_list_type
        }
    
    matching_out_policies = query_result["matching_out_policies"]
    
    # Extract policy names from out-policy matches
    out_policy_names = [match.get("policy_name", "") for match in matching_out_policies]
    
    # out-policy strategy: Pass if matching_out_policies contains original policy
    condition1 = original_policy in out_policy_names
    
    passed = condition1
    
    return {
        "verification_result": "passed" if passed else "failed",
        "reason": f"Out-policy contains '{original_policy}': {condition1}",
        "original_policy": original_policy,
        "category_type": category_type,
        "policy_list_type": policy_list_type,
        "out_policy_matches": out_policy_names,
        "conditions_met": {"out_policy_contains_original": condition1}
    }


def process_single_query(config: dict, company_data: dict, query_data: dict, policy_category: str, max_trials: int, query_index: int, total_queries: int, verbose: bool = False) -> Tuple[dict, int]:
    """Process a single query and return the result along with its index."""
    denied_edge_query = query_data['denied_edge_query']
    
    if verbose:
        print(f"        Query {query_index+1}/{total_queries}: {denied_edge_query}")
        print(f"          📋 Query ID: {query_data['id']}")
        print(f"          🎯 Policy: {policy_category}")
        print(f"          🔄 Attack variation: {query_data['attack_variation']}")
    
    # Call API
    api_result = call_policy_matcher_api(config, company_data, denied_edge_query, max_trials, verbose)
    
    if "error" in api_result:
        result = {
            "query_id": query_data['id'],
            "base_query": query_data['base_query'],
            "denied_edge_query": denied_edge_query,
            "attack_variation": query_data['attack_variation'],
            "verification_result": "error",
            "reason": api_result["error"],
            "original_policy": policy_category,
            "category_type": "out-policy", 
            "policy_list_type": "denylist"
        }
        if verbose:
            print(f"          ❌ Error: {api_result['error']}")
    else:
        # Apply post-processing logic
        result = apply_post_processing_logic(
            api_result, policy_category, "out-policy", "denylist"
        )
        result["query_id"] = query_data['id']
        result["base_query"] = query_data['base_query']
        result["denied_edge_query"] = denied_edge_query
        result["attack_variation"] = query_data['attack_variation']
        result["api_response"] = api_result
        
        if verbose:
            verification_result = result["verification_result"]
            if verification_result == "passed":
                print(f"          ✅ Verification: PASSED")
            else:
                print(f"          ❌ Verification: FAILED")
            print(f"          📝 Reason: {result['reason']}")
            if 'out_policy_matches' in result:
                print(f"          🎯 Out-policy matches: {result['out_policy_matches']}")
    
    return result, query_index


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str, n_proc: int = 1, verbose: bool = False):
    """Processes verification for a specific company."""
    
    # Check if output file already exists
    output_path = os.path.join(output_dir, f'{company_name}.jsonl')
    if os.path.exists(output_path):
        print(f"    ✓ {company_name}.jsonl already exists, skipping")
        return True
    
    # Load company data
    company_data = load_company_data(scenario_dir, company_name)
    
    # Retry settings
    max_trials = config.get('retry', {}).get('max_trials', 3)
    
    verification_results = {
        "company": company_name,
        "out_policy_results": {},
        "summary": {
            "total_queries": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
    }
    
    # Process denied_edge queries (out-policy only)
    denied_edge_queries = company_data['denied_edge_queries']
    
    # Group queries by policy category
    queries_by_policy = {}
    for query_data in denied_edge_queries:
        if query_data['category'] == 'denylist':  # Only process denylist queries
            policy_category = query_data['policy']
            if policy_category not in queries_by_policy:
                queries_by_policy[policy_category] = []
            queries_by_policy[policy_category].append(query_data)
    
    for policy_category, query_data_list in queries_by_policy.items():
        print(f"      Processing denied_edge denylist category: {policy_category}")
        verification_results["out_policy_results"][policy_category] = [None] * len(query_data_list)  # Pre-allocate list for ordering
        total_queries = len(query_data_list)
        
        if n_proc > 1:
            # Parallel processing
            print(f"      Using {n_proc} threads for parallel processing")
            with ThreadPoolExecutor(max_workers=n_proc) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        process_single_query, 
                        config, 
                        company_data, 
                        query_data, 
                        policy_category, 
                        max_trials, 
                        i, 
                        total_queries,
                        verbose
                    ): i 
                    for i, query_data in enumerate(query_data_list)
                }
                
                # Process completed futures
                for future in as_completed(futures):
                    original_index = futures[future]
                    try:
                        result, _ = future.result()
                        verification_results["summary"]["total_queries"] += 1
                        
                        if result["verification_result"] == "error":
                            verification_results["summary"]["errors"] += 1
                        elif result["verification_result"] == "passed":
                            verification_results["summary"]["passed"] += 1
                        else:
                            verification_results["summary"]["failed"] += 1
                        
                        verification_results["out_policy_results"][policy_category][original_index] = result
                    except Exception as e:
                        print(f"        ✗ Error processing query {original_index+1}: {str(e)}")
                        verification_results["summary"]["total_queries"] += 1
                        verification_results["summary"]["errors"] += 1
                        verification_results["out_policy_results"][policy_category][original_index] = {
                            "query_id": query_data_list[original_index]['id'],
                            "base_query": query_data_list[original_index]['base_query'],
                            "denied_edge_query": query_data_list[original_index]['denied_edge_query'],
                            "attack_variation": query_data_list[original_index]['attack_variation'],
                            "verification_result": "error",
                            "reason": f"Processing error: {str(e)}",
                            "original_policy": policy_category,
                            "category_type": "out-policy",
                            "policy_list_type": "denylist"
                        }
        else:
            # Sequential processing (original code)
            for i, query_data in enumerate(query_data_list):
                result, _ = process_single_query(
                    config, 
                    company_data, 
                    query_data, 
                    policy_category, 
                    max_trials, 
                    i, 
                    total_queries,
                    verbose
                )
                verification_results["summary"]["total_queries"] += 1
                
                if result["verification_result"] == "error":
                    verification_results["summary"]["errors"] += 1
                elif result["verification_result"] == "passed":
                    verification_results["summary"]["passed"] += 1
                else:
                    verification_results["summary"]["failed"] += 1
                
                verification_results["out_policy_results"][policy_category][i] = result
    
    # Save results in JSONL format (summary record first, then individual query results)
    summary = verification_results["summary"]
    summary_record = {
        "record_type": "summary",
        "company": company_name,
        **summary
    }

    write_jsonl([summary_record], output_path)

    for policy_category, results_list in verification_results["out_policy_results"].items():
        for result in results_list:
            if result is None:
                continue
            record = {
                "record_type": "result",
                "company": company_name,
                "policy": policy_category,
                **result
            }
            append_jsonl(record, output_path)

    # Print summary
    print(f"    📊 Summary: {summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors out of {summary['total_queries']} queries")
    print(f"    💾 Saved results to {output_path}")
    
    return True


def extract_passed_queries(temp_verification_dir: str, verified_denied_edge_dir: str):
    """Extracts passed queries from temp_verification files and saves them in JSONL format to verified_denied_edge directory."""
    
    # Create verified_denied_edge directory if it doesn't exist
    os.makedirs(verified_denied_edge_dir, exist_ok=True)
    
    # Find all JSONL files in temp_verification directory
    temp_files = glob.glob(os.path.join(temp_verification_dir, '*.jsonl'))
    
    if not temp_files:
        print("No verification files found in temp_verification directory")
        return
    
    print(f"\nProcessing {len(temp_files)} verification files...")
    
    for temp_file in temp_files:
        company_name = os.path.splitext(os.path.basename(temp_file))[0]
        output_path = os.path.join(verified_denied_edge_dir, f'{company_name}.jsonl')
        
        print(f"  Processing {company_name}...")
        
        # Read temp verification data
        try:
            temp_records = read_jsonl(temp_file)
        except Exception as e:
            print(f"    ✗ Error reading {temp_file}: {str(e)}")
            continue

        if not temp_records:
            print(f"    ⚠️ No data found in {temp_file}")
            continue

        # Collect all passed queries as JSONL records
        jsonl_records = []
        policy_counters: Dict[str, int] = {}

        # Note: Only processing out-policy results (denylist) for denied_edge queries
        for record in temp_records:
            if record.get('record_type') != 'result':
                continue

            policy_category = record.get('policy') or record.get('original_policy') or 'unknown'

            if record.get('verification_result') == 'passed':
                policy_counters[policy_category] = policy_counters.get(policy_category, 0) + 1
                counter = policy_counters[policy_category]

                jsonl_record = {
                    "id": record.get('query_id', f"{company_name}-denylist-{policy_category}-{counter}"),
                    "company": company_name,
                    "category": "denylist",
                    "policy": policy_category,
                    "base_query": record.get('base_query', ''),
                    "denied_edge_query": record.get('denied_edge_query', ''),
                    "attack_variation": record.get('attack_variation', '')
                }
                jsonl_records.append(jsonl_record)

        if jsonl_records:
            # Save as JSONL format
            write_jsonl(jsonl_records, output_path)

            # Count queries by category (only denylist for denied_edge queries)
            denylist_count = len(jsonl_records)

            print(f"    ✓ Saved {len(jsonl_records)} passed out-policy queries ({denylist_count} denylist) to {output_path}")
        else:
            print(f"    ⚠️ No passed queries found for {company_name}")
    
    print(f"\n✅ Completed processing verification files")
    print(f"📁 Verified queries saved to: {verified_denied_edge_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies)')
    parser.add_argument('--max_companies', type=int, help='Maximum number of companies to process (used in debug mode)')
    parser.add_argument('--extract_only', action='store_true', help='Only extract passed queries from temp_verification to verified_denied_edge (skip verification)')
    parser.add_argument('--company', nargs='+', help='Company name(s) to process (e.g., AutoViaMotors TelePath)')
    parser.add_argument('--n_proc', type=int, default=1, help='Number of threads for parallel API calls (default: 1)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for detailed logging')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(script_dir, 'config', 'denied_edge_queries_verification.yaml')
    scenario_dir = os.path.join(project_root, 'scenario')
    temp_verification_dir = os.path.join(scenario_dir, 'queries', 'temp_verification_denied_edge')
    verified_denied_edge_dir = os.path.join(scenario_dir, 'queries', 'verified_denied_edge')
    
    # If extract-only mode, just extract passed queries and exit
    if args.extract_only:
        extract_passed_queries(temp_verification_dir, verified_denied_edge_dir)
        return
    
    # Check/create output directory
    os.makedirs(temp_verification_dir, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Debug mode settings
    debug_enabled = args.debug or config.get('debug', {}).get('enabled', False)
    max_companies = args.max_companies or config.get('debug', {}).get('max_companies', 1)
    
    if debug_enabled:
        print(f"🐛 Debug mode enabled - processing maximum {max_companies} companies")
    
    # Get company names list
    print("Getting company names...")
    all_company_names = get_company_names(scenario_dir)
    print(f"Found {len(all_company_names)} companies: {', '.join(all_company_names)}")
    
    # Filter companies based on arguments
    try:
        company_names = filter_company_names(all_company_names, args.company)
        if args.company:
            print(f"Processing specified companies: {', '.join(company_names)}")
    except ValueError as e:
        print(f"❌ Company selection error: {str(e)}")
        print(f"Available companies: {', '.join(all_company_names)}")
        return

    # Limit number of companies in debug mode
    if debug_enabled:
        company_names = company_names[:max_companies]
        print(f"Debug mode: processing {len(company_names)} companies: {', '.join(company_names)}")
    
    # Process each company
    successful = 0
    total = len(company_names)
    
    # Get n_proc and verbose from arguments
    n_proc = args.n_proc
    verbose = args.verbose
    
    if n_proc > 1:
        print(f"\n🚀 Parallel processing enabled with {n_proc} threads")
    if verbose:
        print(f"\n🔍 Verbose mode enabled - detailed logging activated")
    
    for i, company_name in enumerate(company_names, 1):
        print(f"\n[{i}/{total}] Processing {company_name}...")
        if process_company(config, scenario_dir, company_name, temp_verification_dir, n_proc, verbose):
            successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Results saved to: {temp_verification_dir}")
    
    if debug_enabled:
        print(f"🐛 Debug mode was enabled - only processed {total} out of {len(all_company_names)} total companies")
    
    # After verification is complete, extract passed queries to verified_denied_edge
    if successful > 0:
        print("\n📋 Extracting passed queries to verified_denied_edge directory...")
        extract_passed_queries(temp_verification_dir, verified_denied_edge_dir)


if __name__ == "__main__":
    main()
