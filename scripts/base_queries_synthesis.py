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


def load_company_data(scenario_dir: str, company_name: str) -> dict:
    """Loads policy, context, and prompt data for a specific company."""
    # Load policy file
    policy_path = os.path.join(scenario_dir, 'policies', f'{company_name}.json')
    policy_data = read_json(policy_path)
    
    # Load context file
    context_path = os.path.join(scenario_dir, 'contexts', f'{company_name}.txt')
    with open(context_path, 'r', encoding='utf-8') as file:
        context_data = file.read()
    
    return {
        'policy': policy_data,
        'context': context_data
    }


def validate_json_structure(parsed_response: dict, company_data: dict) -> bool:
    """Validates if generated JSON matches expected structure."""
    try:
        # Check required keys
        if "allowlist_test_queries" not in parsed_response:
            return False
        if "denylist_test_queries" not in parsed_response:
            return False
        
        # Check allowlist and denylist categories
        allowlist_categories = set(company_data['policy']['allowlist'].keys())
        denylist_categories = set(company_data['policy']['denylist'].keys())
        
        response_allowlist = set(parsed_response["allowlist_test_queries"].keys())
        response_denylist = set(parsed_response["denylist_test_queries"].keys())
        
        # Check if categories match policy
        if not allowlist_categories.issubset(response_allowlist):
            return False
        if not denylist_categories.issubset(response_denylist):
            return False
        
        # Check if each category has queries
        for category in allowlist_categories:
            if category not in parsed_response["allowlist_test_queries"]:
                return False
            if not isinstance(parsed_response["allowlist_test_queries"][category], list):
                return False
            if len(parsed_response["allowlist_test_queries"][category]) == 0:
                return False
        
        for category in denylist_categories:
            if category not in parsed_response["denylist_test_queries"]:
                return False
            if not isinstance(parsed_response["denylist_test_queries"][category], list):
                return False
            if len(parsed_response["denylist_test_queries"][category]) == 0:
                return False
        
        return True
    except Exception as e:
        print(f"    ⚠️ JSON validation error: {str(e)}")
        return False


def create_messages(config: dict, company_data: dict, company_name: str) -> list:
    """Creates message list for Anthropic API call."""
    # Use new prompt template structure
    prompt_template = config['prompt_template']
    
    # Replace placeholders with actual data
    user_prompt = prompt_template.replace('{company_context}', company_data['context'])
    
    # Convert policy data to JSON string and insert
    import json
    policy_json = json.dumps(company_data['policy'], indent=2, ensure_ascii=False)
    user_prompt = user_prompt.replace('{policy_document}', policy_json)
    
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def process_company(config: dict, scenario_dir: str, company_name: str, output_dir: str):
    """Loads data for a specific company, performs API call, and saves results."""
    
    # Check if output file already exists
    output_path = os.path.join(output_dir, f'{company_name}.json')
    if os.path.exists(output_path):
        print(f"    ✓ {company_name}.json already exists, skipping API call")
        return True
    
    # Load company data
    company_data = load_company_data(scenario_dir, company_name)
    
    # Create messages for API call
    messages = create_messages(config, company_data, company_name)
    
    # Retry settings
    max_trials = config.get('retry', {}).get('max_trials', 3)
    
    for trial in range(1, max_trials + 1):
        print(f"    🔄 Trial {trial}/{max_trials} for {company_name}")
        
        try:
            # Anthropic API call
            response = create_response_chat(
                model=config['anthropic']['model'],
                prompt_input=messages,
                max_completion_tokens=config['anthropic']['max_tokens'],
                temperature=config['anthropic']['temperature'],
                return_type="string"
            )
            
            # Extract only JSON part from response
            try:
                parsed_response = json_style_str_to_dict(response)
                print(f"    ✓ Successfully parsed JSON response for {company_name}")
            except Exception as parse_error:
                print(f"    ⚠️ Failed to parse JSON from response for {company_name}: {str(parse_error)}")
                if trial < max_trials:
                    print(f"    🔄 Retrying due to JSON parsing error...")
                    time.sleep(1)  # Brief wait before retry
                    continue
                else:
                    # Save original response if last attempt also fails
                    parsed_response = {"raw_response": response, "parse_error": str(parse_error)}
            
            # JSON structure validation
            if isinstance(parsed_response, dict) and "raw_response" not in parsed_response:
                is_valid = validate_json_structure(parsed_response, company_data)
                if not is_valid:
                    print(f"    ⚠️ JSON structure validation failed for {company_name}")
                    if trial < max_trials:
                        print(f"    🔄 Retrying due to invalid JSON structure...")
                        time.sleep(1)  # Brief wait before retry
                        continue
                    else:
                        print(f"    ✗ Final attempt failed - saving invalid structure")
            
            # Save results if successfully processed
            output_path = os.path.join(output_dir, f'{company_name}.json')
            write_json(parsed_response, output_path)
            print(f"    ✓ Saved result for {company_name} to {output_path}")
            return True
            
        except Exception as api_error:
            print(f"    ✗ API Error on trial {trial} for {company_name}: {str(api_error)}")
            if trial < max_trials:
                print(f"    🔄 Retrying due to API error...")
                time.sleep(2)  # Longer wait for API errors
                continue
            else:
                print(f"    ✗ All trials failed for {company_name}")
                return False
    
    return False


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Debug mode (process limited number of companies)')
    parser.add_argument('--max-companies', type=int, help='Maximum number of companies to process (used in debug mode)')
    args = parser.parse_args()
    
    # Path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(script_dir, 'config', 'base_queries_synthesis.yaml')
    scenario_dir = os.path.join(project_root, 'scenario')
    output_dir = os.path.join(scenario_dir, 'queries', 'base')
    
    # Check/create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Debug mode settings
    debug_enabled = args.debug or config.get('debug', {}).get('enabled', False)
    max_companies = args.max_companies or config.get('debug', {}).get('max_companies', 3)
    
    if debug_enabled:
        print(f"🐛 Debug mode enabled - processing maximum {max_companies} companies")
    
    # Get company names list
    print("Getting company names...")
    company_names = get_company_names(scenario_dir)
    print(f"Found {len(company_names)} companies: {', '.join(company_names)}")
    
    # Limit number of companies in debug mode
    if debug_enabled:
        company_names = company_names[:max_companies]
        print(f"Debug mode: processing {len(company_names)} companies: {', '.join(company_names)}")
    
    # Process each company
    successful = 0
    total = len(company_names)
    
    for i, company_name in enumerate(company_names, 1):
        print(f"\n[{i}/{total}] Processing {company_name}...")
        if process_company(config, scenario_dir, company_name, output_dir):
            successful += 1
    
    print(f"\n✅ Completed: {successful}/{total} companies processed successfully")
    print(f"📁 Results saved to: {output_dir}")
    
    if debug_enabled:
        print(f"🐛 Debug mode was enabled - only processed {total} out of {len(get_company_names(scenario_dir))} total companies")


if __name__ == "__main__":
    main()
