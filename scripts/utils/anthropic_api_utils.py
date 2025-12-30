import os
from typing import List, Dict

from dotenv import load_dotenv
import anthropic


load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)


def create_response_chat(
    model: str, 
    prompt_input: List[Dict], 
    max_completion_tokens: int,
    temperature: float = 1.0,
    return_type: str = "string"
):
    """
    Generates chat responses using Anthropic Claude API.
    
    Args:
        model: Claude model name (e.g., "claude-3-5-sonnet-20241022")
        prompt_input: Message list [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        max_completion_tokens: Maximum token count
        temperature: Temperature value (0.0-1.0)
        return_type: Return type ("string" or "response")
    
    Returns:
        Response text if string type, full response object if response type
    """

    
    assert return_type in ["string", "str", "text", "txt", "json", "dict", "obj", "object", "response"], \
        "return_type must be one of 'string' ('str', 'text', 'txt') or 'response' ('json', 'dict', 'obj', 'object')"

    # Separate system and user messages
    system_message = ""
    messages = []
    
    for msg in prompt_input:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            messages.append(msg)
    
    # Claude API call
    response = client.messages.create(
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        system=system_message,
        messages=messages
    )

    if return_type.lower() in ["string", "str", "text", "txt"]:
        return response.content[0].text.strip()
    elif return_type.lower() in ["json", "dict", "obj", "object", "response"]:
        return response


def create_response_batch(
    model: str,
    prompt_input_list: List[List[Dict]],
    max_completion_tokens: int,
    temperature: float = 1.0,
    custom_id_list: List[str] = None
):
    """
    Processes multiple prompts in batch.
    
    Args:
        model: Claude model name
        prompt_input_list: List of prompt lists
        max_completion_tokens: Maximum token count
        temperature: Temperature value
        custom_id_list: Custom ID list
    
    Returns:
        Response list
    """
    if custom_id_list is None:
        custom_id_list = [f"request-{i}" for i in range(len(prompt_input_list))]
    
    responses = []
    
    for i, prompt_input in enumerate(prompt_input_list):
        try:
            response = create_response_chat(
                model=model,
                prompt_input=prompt_input,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                return_type="string"
            )
            responses.append({
                "custom_id": custom_id_list[i],
                "response": response,
                "status": "success"
            })
        except Exception as e:
            responses.append({
                "custom_id": custom_id_list[i],
                "response": None,
                "status": "error",
                "error": str(e)
            })
    
    return responses
