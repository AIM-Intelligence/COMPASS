import os
import glob
import argparse
from pathlib import Path
import time
import json

import yaml
import dotenv
from utils.json_utils import read_jsonl, write_jsonl, append_jsonl
from utils.string_utils import json_style_str_to_dict
from utils.vertex_api_utils import create_response_chat

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


def load_company_policy(scenario_dir: str, company_name: str) -> dict:
    """Loads company policy from policies directory."""
    policy_path = os.path.join(scenario_dir, 'policies', f'{company_name}.json')
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    with open(policy_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def load_verified_base_queries(scenario_dir: str, company_name: str) -> list:
    """Loads denylist entries from verified_base directory for a specific company."""
    verified_base_path = os.path.join(scenario_dir, 'queries', 'verified_base', f'{company_name}.jsonl')
    
    if not os.path.exists(verified_base_path):
        raise FileNotFoundError(f"Verified base file not found: {verified_base_path}")
    
    # Read JSONL file
    all_queries = read_jsonl(verified_base_path)
    
    # Filter only denylist entries (NOT allowlist)
    denylist_queries = [query for query in all_queries if query.get('category') == 'denylist']
    
    if not denylist_queries:
        raise ValueError(f"No denylist entries found in {verified_base_path}")
    
    return denylist_queries


def validate_allowed_edge_response(parsed_response: dict) -> bool:
    """Validates if generated allowed_edge response matches expected structure."""
    try:
        # Check required keys (updated to match new config format)
        required_keys = ["subtle_query", "deceptiveness_rating", "why_appears_violative", "why_actually_compliant", "llm_specific_triggers", "false_positive_triggers", "strategies_used"]
        for key in required_keys:
            if key not in parsed_response:
                return False
            if not isinstance(parsed_response[key], str):
                return False
            if len(parsed_response[key].strip()) == 0:
                return False
        
        return True
    except Exception as e:
        print(f"    ⚠️ allowed_edge query JSON validation error: {str(e)}")
        return False


def create_allowed_edge_prompt(config: dict, base_query: str, allowlist_policy: dict, denylist_policy: dict) -> str:
    """Creates prompt for allowed_edge query generation."""
    prompt_template = config['prompt_template']
    
    # Format allowlist policies
    allowlist_text = ""
    for policy_name, policy_desc in allowlist_policy.items():
        allowlist_text += f"    - {policy_name}: {policy_desc}\n"
    
    # Format denylist policies  
    denylist_text = ""
    for policy_name, policy_desc in denylist_policy.items():
        denylist_text += f"    - {policy_name}: {policy_desc}\n"
    
    # Replace placeholders with actual data
    user_prompt = prompt_template.replace('{allowlist_policy_name}', list(allowlist_policy.keys())[0])
    user_prompt = user_prompt.replace('{allowlist_policy_description}', list(allowlist_policy.values())[0])
    user_prompt = user_prompt.replace('{denylist_policy_name}', list(denylist_policy.keys())[0])
    user_prompt = user_prompt.replace('{denylist_policy_description}', list(denylist_policy.values())[0])
    user_prompt = user_prompt.replace('{original_query}', base_query)
    
    return user_prompt


def call_vertex_api(config: dict, prompt: str, max_retries: int = 3) -> str:
    """Makes API call to Vertex AI."""
    vertex_config = config['vertex']
    
    for attempt in range(max_retries):
        try:
            response = create_response_chat(
                model=vertex_config['model'],
                prompt_input=[{"role": "user", "content": prompt}],
                max_completion_tokens=vertex_config['max_tokens'],
                temperature=vertex_config['temperature'],
                region=vertex_config.get('region', 'us-east5'),
                project_id=vertex_config.get('project_id', 'img-sec'),
                return_type="string"
            )
            
            return response
                
        except Exception as e:
            print(f"    ⚠️ API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e


def process_single_query(config: dict, query_item: dict, company_policy: dict) -> list:
    """Processes a single query item to generate allowed_edge variations for each denylist policy."""
    base_query = query_item['base_query']
    policy_name = query_item['policy']
    
    # Get all allowlist policies (allowed_edge queries can appear to be compliant with any of them)
    allowlist_policy = company_policy['allowlist']
    
    # Get the specific denylist policy for this query
    denylist_policy = {policy_name: company_policy['denylist'][policy_name]}
    
    # Create multiple results - one for each denylist policy
    results = []
    base_id = f"Subtle-{query_item['id']}".replace("denylist", "allowlist")
    
    # Try each denylist policy as a potential "appears to violate" target
    for allow_policy_name, allow_policy_desc in allowlist_policy.items():
        # Create prompt for this specific query with this denylist policy
        single_allowlist = {allow_policy_name: allow_policy_desc}
        prompt = create_allowed_edge_prompt(config, base_query, single_allowlist, denylist_policy)
        
        # Get retry settings
        max_trials = config.get('retry', {}).get('max_trials', 3)
        
        for trial in range(1, max_trials + 1):
            try:
                # Call Vertex API
                response = call_vertex_api(config, prompt)
                
                # Parse JSON response
                try:
                    parsed_response = json_style_str_to_dict(response)
                    print(f"      ✓ Successfully parsed allowed_edge response for query {query_item['id']} vs {allow_policy_name}")
                except Exception as parse_error:
                    print(f"      ⚠️ Failed to parse JSON from response: {str(parse_error)}")
                    if trial < max_trials:
                        print(f"      🔄 Retrying due to JSON parsing error...")
                        time.sleep(1)
                        continue
                    else:
                        # Create error result for this allowlist policy and continue to next
                        result = query_item.copy()
                        result['id'] = f"{base_id}-{allow_policy_name}-parse_error"
                        result['allowlist_variation'] = allow_policy_name
                        result['allowed_edge_query'] = f"Error: {str(parse_error)}"
                        result['deceptiveness_rating'] = "N/A"
                        result['why_appears_violative'] = "N/A"
                        result['why_actually_compliant'] = "N/A"
                        result['llm_specific_triggers'] = "N/A"
                        result['false_positive_triggers'] = "N/A"
                        result['strategies_used'] = "N/A"
                        result['raw_response'] = response
                        results.append(result)
                        break
                
                # Validate response structure
                if isinstance(parsed_response, dict) and validate_allowed_edge_response(parsed_response):
                    # Create successful result
                    result = query_item.copy()
                    result['id'] = f"{base_id}-{allow_policy_name}"
                    result['allowlist_variation'] = allow_policy_name
                    result['allowed_edge_query'] = parsed_response['subtle_query']
                    result['deceptiveness_rating'] = parsed_response['deceptiveness_rating']
                    result['why_appears_violative'] = parsed_response['why_appears_violative']
                    result['why_actually_compliant'] = parsed_response['why_actually_compliant']
                    result['llm_specific_triggers'] = parsed_response['llm_specific_triggers']
                    result['false_positive_triggers'] = parsed_response['false_positive_triggers']
                    result['strategies_used'] = parsed_response['strategies_used']
                    results.append(result)
                    break
                else:
                    print(f"      ⚠️ allowed_edge response validation failed")
                    if trial < max_trials:
                        print(f"      🔄 Retrying due to invalid response structure...")
                        time.sleep(1)
                        continue
                    else:
                        # Create error result for this allowlist policy and continue to next
                        result = query_item.copy()
                        result['id'] = f"{base_id}-{allow_policy_name}-validation_error"
                        result['allowlist_variation'] = allow_policy_name
                        result['allowed_edge_query'] = "Error: Invalid response structure"
                        result['deceptiveness_rating'] = "N/A"
                        result['why_appears_violative'] = "N/A"
                        result['why_actually_compliant'] = "N/A"
                        result['llm_specific_triggers'] = "N/A"
                        result['false_positive_triggers'] = "N/A"
                        result['strategies_used'] = "N/A"
                        results.append(result)
                        break
                        
            except Exception as e:
                print(f"      ✗ Error on trial {trial}: {str(e)}")
                if trial < max_trials:
                    print(f"      🔄 Retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"      ✗ All trials failed for query {query_item['id']} vs {allow_policy_name}")
                    # Create error result for this allowlist policy and continue to next
                    result = query_item.copy()
                    result['id'] = f"{base_id}-{allow_policy_name}-api_error"
                    result['allowlist_variation'] = allow_policy_name
                    result['allowed_edge_query'] = f"Error: {str(e)}"
                    result['deceptiveness_rating'] = "N/A"
                    result['why_appears_violative'] = "N/A"
                    result['why_actually_compliant'] = "N/A"
                    result['llm_specific_triggers'] = "N/A"
                    result['false_positive_triggers'] = "N/A"
                    result['strategies_used'] = "N/A"
                    results.append(result)
                    break
    
    # If no results were generated at all, create a single error result
    if not results:
        result = query_item.copy()
        result['id'] = f"{base_id}-error"
        result['allowlist_variation'] = "error"
        result['allowed_edge_query'] = "Error: No allowed_edge queries generated"
        result['deceptiveness_rating'] = "N/A"
        result['why_appears_violative'] = "N/A"
        result['why_actually_compliant'] = "N/A"
        result['llm_specific_triggers'] = "N/A"
        result['false_positive_triggers'] = "N/A"
        result['strategies_used'] = "N/A"
        results.append(result)
    
    return results


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str, debug_limit: int = None):
    """Processes all allowlist queries for a specific company."""
    
    # Setup output path
    output_path = os.path.join(output_dir, f'{company_name}.jsonl')
    
    # Remove existing file to start fresh (or you could implement resume logic)
    if os.path.exists(output_path):
        print(f"    🔄 Removing existing {company_name}.jsonl to start fresh")
        os.remove(output_path)
    
    try:
        # Load company policy
        company_policy = load_company_policy(scenario_dir, company_name)
        
        # Load denylist queries from verified_base  
        denylist_queries = load_verified_base_queries(scenario_dir, company_name)
        print(f"    📋 Loaded {len(denylist_queries)} denylist queries")
        
        total_processed = 0
        
        # Apply debug limit if specified
        queries_to_process = denylist_queries
        if debug_limit:
            queries_to_process = denylist_queries[:debug_limit]
            print(f"    🐛 Debug mode: limiting to {len(queries_to_process)} queries")
        
        # Process each denylist query
        for i, query_item in enumerate(queries_to_process, 1):
            print(f"      [{i}/{len(queries_to_process)}] Processing: {query_item['base_query'][:80]}...")
            
            try:
                result_list = process_single_query(config, query_item, company_policy)
                if result_list:
                    # Immediately append each result to JSONL file
                    for result in result_list:
                        append_jsonl(result, output_path)
                    total_processed += len(result_list)
                    print(f"      ✓ Generated {len(result_list)} variations for query {query_item['id']} and saved to file")
                else:
                    print(f"      ⚠️ No results generated for query {query_item['id']}")
            except Exception as query_error:
                print(f"      ✗ Critical error processing query {query_item['id']}: {query_error}")
                # Create error result to maintain consistency
                error_result = query_item.copy()
                error_result['id'] = f"Critical-Error-{query_item['id']}"
                error_result['allowlist_variation'] = "critical_error"
                error_result['allowed_edge_query'] = f"Critical Error: {str(query_error)}"
                error_result['deceptiveness_rating'] = "N/A"
                error_result['why_appears_violative'] = "N/A"
                error_result['why_actually_compliant'] = "N/A"
                error_result['llm_specific_triggers'] = "N/A"
                error_result['false_positive_triggers'] = "N/A"
                error_result['strategies_used'] = "N/A"
                # Immediately append error result to JSONL file
                append_jsonl(error_result, output_path)
                total_processed += 1
        
        print(f"    ✅ Processed {total_processed} allowed_edge query variations for {company_name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {company_name}: {e}")
        # Results are already saved incrementally, so no need for additional save logic
        return False


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate allowed_edge variations of allowlist queries based on denylist queries')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies and queries)')
    parser.add_argument('--max-companies', type=int, help='Maximum number of companies to process (used in debug mode)')
    parser.add_argument('--queries-per-company', type=int, help='Maximum queries per company in debug mode')
    parser.add_argument('--company', type=str, help='Specific company name to process (if not specified, processes all companies)')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(script_dir, 'config', 'allowed_edge_queries_synthesis.yaml')
    scenario_dir = os.path.join(project_root, 'scenario')
    output_dir = os.path.join(scenario_dir, 'queries', 'allowed_edge')
    
    # Check/create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(config_path)
    
    # Debug mode settings
    debug_enabled = args.debug or config.get('debug', {}).get('enabled', False)
    max_companies = args.max_companies or config.get('debug', {}).get('max_companies', 1)
    queries_per_company = args.queries_per_company or (2 if debug_enabled else None)
    
    if debug_enabled:
        print(f"🐛 Debug mode enabled - processing maximum {max_companies} companies")
        if queries_per_company:
            print(f"🐛 Debug mode: limiting to {queries_per_company} queries per company")
    
    # Check API key (for Vertex API)
    if config['vertex']['model'] == "gemini-2.5-pro":
        if not os.getenv('VERTEX_API_KEY'):
            print("❌ ERROR: VERTEX_API_KEY environment variable not set")
            print("Please set your Vertex API key in the .env file or environment")
            return
    # Claude doesn't need API key check as it uses project credentials
    
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
        if process_company(config, scenario_dir, company_name, output_dir, queries_per_company):
            successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Results saved to: {output_dir}")
    
    if debug_enabled:
        print(f"🐛 Debug mode was enabled - processed {total} out of {len(get_company_names(scenario_dir))} total companies")
        if queries_per_company:
            print(f"🐛 Limited to {queries_per_company} queries per company")


if __name__ == "__main__":
    main()