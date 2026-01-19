import os
import glob
import argparse
from pathlib import Path
import time

import yaml
from utils.json_utils import read_json, write_json
from utils.anthropic_api_utils import create_response_chat
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
    """Loads policy, context, and base queries data for a specific company."""
    # Load policy file
    policy_path = os.path.join(scenario_dir, 'policies', f'{company_name}.json')
    policy_data = read_json(policy_path)
    
    # Load context file
    context_path = os.path.join(scenario_dir, 'contexts', f'{company_name}.txt')
    with open(context_path, 'r', encoding='utf-8') as file:
        context_data = file.read()
    
    # Load base queries file
    base_queries_path = os.path.join(scenario_dir, 'queries', 'base', f'{company_name}.json')
    base_queries_data = read_json(base_queries_path)
    
    return {
        'policy': policy_data,
        'context': context_data,
        'base_queries': base_queries_data
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


def call_policy_matcher_api(config: dict, company_data: dict, query: str, max_trials: int = 3) -> dict:
    """Calls the policy matcher API for a single query."""
    messages = create_messages(config, company_data, query)
    
    for trial in range(1, max_trials + 1):
        try:
            response = create_response_chat(
                model=config['anthropic']['model'],
                prompt_input=messages,
                max_completion_tokens=config['anthropic']['max_tokens'],
                temperature=config['anthropic']['temperature'],
                return_type="string"
            )
            
            # Parse JSON response
            try:
                parsed_response = json_style_str_to_dict(response)
                return parsed_response
            except Exception as parse_error:
                print(f"        ⚠️ JSON parsing failed on trial {trial}: {str(parse_error)}")
                if trial < max_trials:
                    time.sleep(1)
                    continue
                else:
                    return {"error": f"JSON parsing failed: {str(parse_error)}", "raw_response": response}
                    
        except Exception as api_error:
            print(f"        ✗ API error on trial {trial}: {str(api_error)}")
            if trial < max_trials:
                time.sleep(2)
                continue
            else:
                return {"error": f"API error: {str(api_error)}"}
    
    return {"error": "All trials failed"}


def apply_post_processing_logic(query_result: dict, original_policy: str, category_type: str, policy_list_type: str) -> dict:
    """
    Applies post-processing logic based on category type.
    
    Args:
        query_result: API response with matching_policies
        original_policy: The original policy name from base queries
        category_type: 'in-policy' or 'out-policy'
        policy_list_type: 'allowlist' or 'denylist'
    
    Returns:
        dict with verification result
    """
    if "matching_policies" not in query_result:
        return {
            "verification_result": "failed",
            "reason": "No matching_policies found in API response",
            "original_policy": original_policy,
            "category_type": category_type,
            "policy_list_type": policy_list_type
        }
    
    matching_policies = query_result["matching_policies"]
    allowlist_matches = matching_policies.get("allowlist_matches", [])
    denylist_matches = matching_policies.get("denylist_matches", [])
    
    # Extract policy names from matches
    allowlist_policy_names = [match.get("policy_name", "") for match in allowlist_matches]
    denylist_policy_names = [match.get("policy_name", "") for match in denylist_matches]
    
    if category_type == "in-policy":
        # in-policy strategy: Pass if (1) allowlist contains original policy AND (2) denylist is empty
        condition1 = original_policy in allowlist_policy_names
        condition2 = len(denylist_matches) == 0
        
        passed = condition1 and condition2
        
        return {
            "verification_result": "passed" if passed else "failed",
            "reason": f"Condition1 (allowlist contains '{original_policy}'): {condition1}, Condition2 (denylist empty): {condition2}",
            "original_policy": original_policy,
            "category_type": category_type,
            "policy_list_type": policy_list_type,
            "allowlist_matches": allowlist_policy_names,
            "denylist_matches": denylist_policy_names,
            "conditions_met": {"allowlist_contains_original": condition1, "denylist_empty": condition2}
        }
    
    elif category_type == "out-policy":
        # out-policy strategy: Pass if denylist contains original policy (ignore allowlist)
        condition1 = original_policy in denylist_policy_names
        
        passed = condition1
        
        return {
            "verification_result": "passed" if passed else "failed",
            "reason": f"Condition1 (denylist contains '{original_policy}'): {condition1}",
            "original_policy": original_policy,
            "category_type": category_type,
            "policy_list_type": policy_list_type,
            "allowlist_matches": allowlist_policy_names,
            "denylist_matches": denylist_policy_names,
            "conditions_met": {"denylist_contains_original": condition1}
        }
    
    else:
        return {
            "verification_result": "failed",
            "reason": f"Unknown category_type: {category_type}",
            "original_policy": original_policy,
            "category_type": category_type,
            "policy_list_type": policy_list_type
        }


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str):
    """Processes verification for a specific company."""
    
    # Check if output file already exists
    output_path = os.path.join(output_dir, f'{company_name}.json')
    if os.path.exists(output_path):
        print(f"    ✓ {company_name}.json already exists, skipping")
        return True
    
    # Load company data
    company_data = load_company_data(scenario_dir, company_name)
    
    # Retry settings
    max_trials = config.get('retry', {}).get('max_trials', 3)
    
    verification_results = {
        "company": company_name,
        "in_policy_results": {},
        "out_policy_results": {},
        "summary": {
            "total_queries": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
    }
    
    # Process allowlist queries (in-policy)
    allowlist_queries = company_data['base_queries'].get('allowlist_test_queries', {})
    for policy_category, queries in allowlist_queries.items():
        print(f"      Processing allowlist category: {policy_category}")
        verification_results["in_policy_results"][policy_category] = []
        
        for i, query in enumerate(queries):
            print(f"        Query {i+1}/{len(queries)}: {query[:50]}...")
            verification_results["summary"]["total_queries"] += 1
            
            # Call API
            api_result = call_policy_matcher_api(config, company_data, query, max_trials)
            
            if "error" in api_result:
                result = {
                    "query": query,
                    "verification_result": "error",
                    "reason": api_result["error"],
                    "original_policy": policy_category,
                    "category_type": "in-policy",
                    "policy_list_type": "allowlist"
                }
                verification_results["summary"]["errors"] += 1
            else:
                # Apply post-processing logic
                result = apply_post_processing_logic(
                    api_result, policy_category, "in-policy", "allowlist"
                )
                result["query"] = query
                result["api_response"] = api_result
                
                if result["verification_result"] == "passed":
                    verification_results["summary"]["passed"] += 1
                else:
                    verification_results["summary"]["failed"] += 1
            
            verification_results["in_policy_results"][policy_category].append(result)
    
    # Process denylist queries (out-policy)  
    denylist_queries = company_data['base_queries'].get('denylist_test_queries', {})
    for policy_category, queries in denylist_queries.items():
        print(f"      Processing denylist category: {policy_category}")
        verification_results["out_policy_results"][policy_category] = []
        
        for i, query in enumerate(queries):
            print(f"        Query {i+1}/{len(queries)}: {query[:50]}...")
            verification_results["summary"]["total_queries"] += 1
            
            # Call API
            api_result = call_policy_matcher_api(config, company_data, query, max_trials)
            
            if "error" in api_result:
                result = {
                    "query": query,
                    "verification_result": "error",
                    "reason": api_result["error"],
                    "original_policy": policy_category,
                    "category_type": "out-policy", 
                    "policy_list_type": "denylist"
                }
                verification_results["summary"]["errors"] += 1
            else:
                # Apply post-processing logic
                result = apply_post_processing_logic(
                    api_result, policy_category, "out-policy", "denylist"
                )
                result["query"] = query
                result["api_response"] = api_result
                
                if result["verification_result"] == "passed":
                    verification_results["summary"]["passed"] += 1
                else:
                    verification_results["summary"]["failed"] += 1
            
            verification_results["out_policy_results"][policy_category].append(result)
    
    # Save results
    write_json(verification_results, output_path)
    
    # Print summary
    summary = verification_results["summary"]
    print(f"    📊 Summary: {summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors out of {summary['total_queries']} queries")
    print(f"    💾 Saved results to {output_path}")
    
    return True


def extract_passed_queries(temp_verification_dir: str, verified_base_dir: str):
    """Extracts passed queries from temp_verification files and saves them in JSONL format to verified_base directory."""
    
    # Create verified_base directory if it doesn't exist
    os.makedirs(verified_base_dir, exist_ok=True)
    
    # Find all JSON files in temp_verification directory
    temp_files = glob.glob(os.path.join(temp_verification_dir, '*.json'))
    
    if not temp_files:
        print("No verification files found in temp_verification directory")
        return
    
    print(f"\nProcessing {len(temp_files)} verification files...")
    
    for temp_file in temp_files:
        company_name = os.path.splitext(os.path.basename(temp_file))[0]
        output_path = os.path.join(verified_base_dir, f'{company_name}.jsonl')
        
        print(f"  Processing {company_name}...")
        
        # Read temp verification data
        try:
            temp_data = read_json(temp_file)
        except Exception as e:
            print(f"    ✗ Error reading {temp_file}: {str(e)}")
            continue
        
        # Collect all passed queries as JSONL records
        jsonl_records = []
        
        # Extract passed queries from in_policy_results (allowlist)
        in_policy_results = temp_data.get('in_policy_results', {})
        for policy_category, results in in_policy_results.items():
            query_counter = 1
            for result in results:
                if result.get('verification_result') == 'passed':
                    record = {
                        "id": f"{company_name}-allowlist-{policy_category}-{query_counter}",
                        "company": company_name,
                        "category": "allowlist",
                        "policy": policy_category,
                        "base_query": result.get('query', '')
                    }
                    jsonl_records.append(record)
                    query_counter += 1
        
        # Extract passed queries from out_policy_results (denylist)
        out_policy_results = temp_data.get('out_policy_results', {})
        for policy_category, results in out_policy_results.items():
            query_counter = 1
            for result in results:
                if result.get('verification_result') == 'passed':
                    record = {
                        "id": f"{company_name}-denylist-{policy_category}-{query_counter}",
                        "company": company_name,
                        "category": "denylist",
                        "policy": policy_category,
                        "base_query": result.get('query', '')
                    }
                    jsonl_records.append(record)
                    query_counter += 1
        
        if jsonl_records:
            # Save as JSONL format
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in jsonl_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Count queries by category
            allowlist_count = sum(1 for r in jsonl_records if r['category'] == 'allowlist')
            denylist_count = sum(1 for r in jsonl_records if r['category'] == 'denylist')
            
            print(f"    ✓ Saved {len(jsonl_records)} passed queries ({allowlist_count} allowlist, {denylist_count} denylist) to {output_path}")
        else:
            print(f"    ⚠️ No passed queries found for {company_name}")
    
    print(f"\n✅ Completed processing verification files")
    print(f"📁 Verified queries saved to: {verified_base_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--company', nargs='+', help='Company name(s) to process (without extension)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies)')
    parser.add_argument('--max-companies', type=int, help='Maximum number of companies to process (used in debug mode)')
    parser.add_argument('--extract-only', action='store_true', help='Only extract passed queries from temp_verification to verified_base (skip verification)')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(script_dir, 'config', 'base_queries_verification.yaml')
    scenario_dir = os.path.join(project_root, 'scenario')
    temp_verification_dir = os.path.join(scenario_dir, 'queries', 'temp_verification')
    verified_base_dir = os.path.join(scenario_dir, 'queries', 'verified_base')
    
    # If extract-only mode, just extract passed queries and exit
    if args.extract_only:
        extract_passed_queries(temp_verification_dir, verified_base_dir)
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
    company_names = get_company_names(scenario_dir)
    print(f"Found {len(company_names)} companies: {', '.join(company_names)}")

    # Filter companies if specified
    if args.company:
        try:
            company_names = filter_company_names(company_names, args.company)
            print(f"Selected companies: {', '.join(company_names)}")
        except ValueError as e:
            print(f"✗ {str(e)}")
            return
    
    # Limit number of companies in debug mode
    if debug_enabled:
        company_names = company_names[:max_companies]
        print(f"Debug mode: processing {len(company_names)} companies: {', '.join(company_names)}")
    
    # Process each company
    successful = 0
    total = len(company_names)
    
    for i, company_name in enumerate(company_names, 1):
        print(f"\n[{i}/{total}] Processing {company_name}...")
        if process_company(config, scenario_dir, company_name, temp_verification_dir):
            successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Results saved to: {temp_verification_dir}")
    
    if debug_enabled:
        print(f"🐛 Debug mode was enabled - only processed {total} out of {len(get_company_names(scenario_dir))} total companies")
    
    # After verification is complete, extract passed queries to verified_base
    if successful > 0:
        print("\n📋 Extracting passed queries to verified_base directory...")
        extract_passed_queries(temp_verification_dir, verified_base_dir)


if __name__ == "__main__":
    main()
