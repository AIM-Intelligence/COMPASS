import os
import glob
import json
import argparse
import time
from typing import Dict, List, Optional, Any
from multiprocessing import Pool
from functools import partial

import yaml
from tqdm import tqdm
from utils.json_utils import read_jsonl, write_jsonl, append_jsonl
from utils.unified_api_utils import create_response_chat_for_structured_output, get_provider_from_config, check_api_key, get_required_env_var


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def get_company_names(allowed_edge_dir: str) -> list:
    """Gets list of company names from allowed_edge queries directory."""
    allowed_edge_files = glob.glob(os.path.join(allowed_edge_dir, '*.jsonl'))
    company_names = [os.path.splitext(os.path.basename(f))[0] for f in allowed_edge_files]
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
    """Loads policy and allowed_edge queries data for a specific company."""
    # Load policy file
    policy_path = os.path.join(scenario_dir, 'policies', f'{company_name}.json')
    with open(policy_path, 'r', encoding='utf-8') as f:
        policy_data = json.load(f)
    
    # Load allowed_edge queries file
    allowed_edge_queries_path = os.path.join(scenario_dir, 'queries', 'allowed_edge', f'{company_name}.jsonl')
    allowed_edge_queries_data = read_jsonl(allowed_edge_queries_path)
    
    return {
        'policy': policy_data,
        'allowed_edge_queries': allowed_edge_queries_data
    }


def create_messages(config: dict, policy_data: dict, allowed_edge_query_item: dict) -> list:
    """Creates message list for OpenAI API call."""
    prompt_template = config['prompt_template']
    
    # Replace placeholders with actual data
    user_prompt = prompt_template.replace('{subtle_query}', allowed_edge_query_item.get('allowed_edge_query', allowed_edge_query_item.get('subtle_query', '')))
    user_prompt = user_prompt.replace('{json_style_policies}', json.dumps(policy_data, indent=2, ensure_ascii=False))
    user_prompt = user_prompt.replace('{why_appears_violative}', allowed_edge_query_item.get('why_appears_violative', ''))
    user_prompt = user_prompt.replace('{why_actually_compliant}', allowed_edge_query_item.get('why_actually_compliant', ''))
    user_prompt = user_prompt.replace('{llm_specific_triggers}', allowed_edge_query_item.get('llm_specific_triggers', ''))
    user_prompt = user_prompt.replace('{false_positive_triggers}', allowed_edge_query_item.get('false_positive_triggers', ''))
    user_prompt = user_prompt.replace('{strategies_used}', allowed_edge_query_item.get('strategies_used', ''))
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def call_verification_api(config: dict, policy_data: dict, allowed_edge_query_item: dict, max_trials: int = 3) -> Optional[dict]:
    """Calls the API to verify a single allowed_edge query (structured output, OpenAI only)."""
    messages = create_messages(config, policy_data, allowed_edge_query_item)
    
    # Prepare the response schema for structured output
    response_schema = {
        "type": "json_schema",
        "json_schema": config['output_json_schema']
    }
    
    for trial in range(1, max_trials + 1):
        try:
            # Structured output API call (provider selected automatically from config)
            response = create_response_chat_for_structured_output(
                config=config,
                prompt_input=messages,
                response_schema=response_schema,
                return_type="dict"
            )
            
            if response is None:
                print(f"        ⚠️ Refusal on trial {trial}")
                if trial < max_trials:
                    time.sleep(1)
                    continue
                else:
                    return None
            
            return response
            
        except Exception as api_error:
            print(f"        ✗ API error on trial {trial}: {str(api_error)}")
            if trial < max_trials:
                time.sleep(2)
                continue
            else:
                return None
    
    return None


def process_single_query(args: tuple) -> dict:
    """Process a single query with verification API."""
    config, policy_data, allowed_edge_query_item, max_trials, company_name, i = args
    
    # Call verification API
    api_result = call_verification_api(
        config,
        policy_data,
        allowed_edge_query_item,
        max_trials
    )
    
    if api_result is None:
        # API call failed
        result = {
            "id": allowed_edge_query_item.get('id', f"{company_name}-{i}"),
            "company": company_name,
            "category": "allowlist",
            "policy": allowed_edge_query_item.get('policy', ''),
            "base_query": allowed_edge_query_item.get('base_query', ''),
            "allowed_edge_query": allowed_edge_query_item.get('allowed_edge_query', allowed_edge_query_item.get('subtle_query', '')),
            "verification_status": "error",
            "final_verdict": None,
            "confidence_level": None,
            "error": "API call failed after all retries"
        }
    else:
        # API call succeeded - using simplified structure without perspective classification
        final_verdict = api_result.get('final_verdict', 'UNKNOWN')
        confidence_level = api_result.get('confidence_level', 'UNKNOWN')
        initial_impression = api_result.get('initial_impression', '')
        suspicious_elements = api_result.get('suspicious_elements', []) or []
        deeper_analysis = api_result.get('deeper_analysis', '')
        policy_evaluation = api_result.get('policy_evaluation', {}) or {}
        claim_verification = api_result.get('claim_verification', {}) or {}
        verdict_justification = api_result.get('verdict_justification', '')
        potential_errors = api_result.get('potential_errors', '')

        result = {
            "id": allowed_edge_query_item.get('id', f"{company_name}-{i}"),
            "company": company_name,
            "category": "allowlist",
            "policy": allowed_edge_query_item.get('policy', ''),
            "base_query": allowed_edge_query_item.get('base_query', ''),
            "allowed_edge_query": allowed_edge_query_item.get('allowed_edge_query', allowed_edge_query_item.get('subtle_query', '')),
            "verification_status": "completed",
            "final_verdict": final_verdict,
            "confidence_level": confidence_level,
            "initial_impression": initial_impression,
            "suspicious_elements": suspicious_elements,
            "deeper_analysis": deeper_analysis,
            "policy_evaluation": policy_evaluation,
            "claim_verification": claim_verification,
            "verdict_justification": verdict_justification,
            "potential_errors": potential_errors,
            "deceptiveness_rating": allowed_edge_query_item.get('deceptiveness_rating', ''),
            "why_appears_violative": allowed_edge_query_item.get('why_appears_violative', ''),
            "why_actually_compliant": allowed_edge_query_item.get('why_actually_compliant', ''),
            "strategies_used": allowed_edge_query_item.get('strategies_used', '')
        }
    
    return result


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str, n_proc: int = 1):
    """Processes verification for a specific company."""
    
    # Check if output file already exists
    output_path = os.path.join(output_dir, f'{company_name}.jsonl')
    if os.path.exists(output_path):
        print(f"    ✓ {company_name}.jsonl already exists, skipping")
        return True
    
    # Load company data
    try:
        company_data = load_company_data(scenario_dir, company_name)
    except Exception as e:
        print(f"    ✗ Error loading data for {company_name}: {str(e)}")
        raise e
        return False
    
    # Retry settings
    max_trials = config.get('retry', {}).get('max_trials', 10)
    
    total_queries = len(company_data['allowed_edge_queries'])
    
    # Statistics
    stats = {
        'total': total_queries,
        'in_policy': 0,
        'out_policy': 0,
        'errors': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0
    }
    
    print(f"    Processing {total_queries} allowed_edge queries with {n_proc} parallel workers...")
    
    if n_proc > 1:
        # Prepare arguments for parallel processing
        args_list = [
            (config, company_data['policy'], allowed_edge_query_item, max_trials, company_name, i)
            for i, allowed_edge_query_item in enumerate(company_data['allowed_edge_queries'], 1)
        ]
        
        # Process queries in parallel
        with Pool(processes=n_proc) as pool:
            results = list(tqdm(
                pool.imap(process_single_query, args_list),
                total=total_queries,
                desc=f"      {company_name}",
                leave=False
            ))
        
        # Save results and update statistics
        for i, result in enumerate(results, 1):
            # Update statistics
            if result['verification_status'] == 'error':
                stats['errors'] += 1
            else:
                final_verdict = result['final_verdict']
                confidence_level = result['confidence_level']
                
                if final_verdict == 'IN-POLICY':
                    stats['in_policy'] += 1
                elif final_verdict == 'OUT-OF-POLICY':
                    stats['out_policy'] += 1
                
                if confidence_level == 'HIGH':
                    stats['high_confidence'] += 1
                elif confidence_level == 'MEDIUM':
                    stats['medium_confidence'] += 1
                elif confidence_level == 'LOW':
                    stats['low_confidence'] += 1
            
            # Save result immediately to file
            append_jsonl(result, output_path)
            
            # Print progress every 100 queries
            if i % 100 == 0:
                print(f"        Progress: {i}/{total_queries} queries processed")
    else:
        # Process each allowed_edge query sequentially (original logic)
        for i, allowed_edge_query_item in enumerate(tqdm(company_data['allowed_edge_queries'], desc=f"      {company_name}", leave=False), 1):
            result = process_single_query(
                (config, company_data['policy'], allowed_edge_query_item, max_trials, company_name, i)
            )
            
            # Update statistics
            if result['verification_status'] == 'error':
                stats['errors'] += 1
            else:
                final_verdict = result['final_verdict']
                confidence_level = result['confidence_level']
                
                if final_verdict == 'IN-POLICY':
                    stats['in_policy'] += 1
                elif final_verdict == 'OUT-OF-POLICY':
                    stats['out_policy'] += 1
                
                if confidence_level == 'HIGH':
                    stats['high_confidence'] += 1
                elif confidence_level == 'MEDIUM':
                    stats['medium_confidence'] += 1
                elif confidence_level == 'LOW':
                    stats['low_confidence'] += 1
            
            # Save result immediately to file
            append_jsonl(result, output_path)
            
            # Print progress every 100 queries
            if i % 100 == 0:
                print(f"        Progress: {i}/{total_queries} queries processed")
    
    # Print summary
    print(f"    📊 Summary:")
    print(f"       - Total: {stats['total']} queries")
    print(f"       - IN-POLICY: {stats['in_policy']} ({stats['in_policy']*100/stats['total']:.1f}%)")
    print(f"       - OUT-OF-POLICY: {stats['out_policy']} ({stats['out_policy']*100/stats['total']:.1f}%)")
    print(f"       - Errors: {stats['errors']} ({stats['errors']*100/stats['total']:.1f}%)")
    print(f"       - Confidence: HIGH={stats['high_confidence']}, MEDIUM={stats['medium_confidence']}, LOW={stats['low_confidence']}")
    print(f"    💾 Saved results to {output_path}")
    
    return True


def post_process_verification_results(temp_dir: str, output_dir: str) -> None:
    """Filters completed in-policy queries and saves trimmed records."""
    temp_files = sorted(glob.glob(os.path.join(temp_dir, '*.jsonl')))

    if not temp_files:
        print("    ⚠️ No temporary verification files found for post-processing")
        return

    for temp_file in temp_files:
        records = read_jsonl(temp_file)
        filtered_records = []

        for record in records:
            # Now just filtering for IN-POLICY queries without perspective check
            if (
                record.get('verification_status') == 'completed'
                and record.get('final_verdict') == 'IN-POLICY'
            ):
                filtered_records.append({
                    "id": record.get('id'),
                    "company": record.get('company'),
                    "category": record.get('category'),
                    "policy": record.get('policy'),
                    "base_query": record.get('base_query'),
                    "allowed_edge_query": record.get('allowed_edge_query', record.get('subtle_query')),
                })

        output_path = os.path.join(output_dir, os.path.basename(temp_file))
        write_jsonl(filtered_records, output_path)

        print(
            f"    ✓ Post-processed {os.path.basename(temp_file)}: "
            f"{len(filtered_records)} entries saved"
        )


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Verify allowed_edge queries using OpenAI API')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies)')
    parser.add_argument('--max_companies', type=int, help='Maximum number of companies to process')
    parser.add_argument('--company', nargs='+', help='Company name(s) to process')
    parser.add_argument('--n_proc', type=int, default=50, help='Number of parallel processes for API calls (default: 1)')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(script_dir, 'config', 'allowed_edge_queries_verification.yaml')
    scenario_dir = os.path.join(project_root, 'scenario')
    allowed_edge_dir = os.path.join(scenario_dir, 'queries', 'allowed_edge')
    temp_verified_dir = os.path.join(scenario_dir, 'queries', 'temp_verification_allowed_edge')
    verified_output_dir = os.path.join(scenario_dir, 'queries', 'verified_allowed_edge')

    # Check/create output directories
    os.makedirs(temp_verified_dir, exist_ok=True)
    os.makedirs(verified_output_dir, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Check API key
    provider = get_provider_from_config(config)
    print(f"📡 Using API provider: {provider}")
    if not check_api_key(config):
        print(f"❌ ERROR: {get_required_env_var(provider)} environment variable not set")
        return
    
    # Debug mode settings
    debug_enabled = args.debug or config.get('debug', {}).get('enabled', False)
    max_companies = args.max_companies or config.get('debug', {}).get('max_companies', 1)
    
    if debug_enabled:
        print(f"🐛 Debug mode enabled - processing maximum {max_companies} companies")
    
    # Get company names list
    print("Getting company names...")
    company_names = get_company_names(allowed_edge_dir)

    try:
        company_names = filter_company_names(company_names, args.company)
        if args.company:
            print(f"Processing selected companies: {', '.join(company_names)}")
        else:
            print(f"Found {len(company_names)} companies: {', '.join(company_names)}")
    except ValueError as e:
        print(f"Company selection error: {str(e)}")
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
        if process_company(config, scenario_dir, company_name, temp_verified_dir, n_proc=args.n_proc):
            successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Raw verification results saved to: {temp_verified_dir}")

    print("\n🧹 Running post-processing to generate verified JSONL outputs...")
    post_process_verification_results(temp_verified_dir, verified_output_dir)
    print(f"📁 Post-processed results saved to: {verified_output_dir}")
    
    if debug_enabled and not args.company:
        print(f"🐛 Debug mode was enabled - only processed {total} out of {len(get_company_names(allowed_edge_dir))} total companies")


if __name__ == "__main__":
    main()
