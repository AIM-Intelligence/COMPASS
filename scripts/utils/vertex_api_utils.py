import os
from typing import List, Dict
import json

from dotenv import load_dotenv
from anthropic import AnthropicVertex
from google import genai
from tqdm import tqdm
from .json_utils import write_jsonl


load_dotenv()


def create_response_chat(
    model: str, 
    prompt_input: List[Dict], 
    max_completion_tokens: int,
    temperature: float = 1,
    region: str = "us-east5",
    project_id: str = "",
    return_type: str = "string"
):
    """
    Create response using Vertex AI - supports both Claude and Gemini models
    """
    assert return_type in ["string", "str", "text", "txt", "json", "dict", "obj", "object", "response"], \
        "return_type must be one of 'string' ('str', 'text', 'txt') or 'response' ('json', 'dict', 'obj', 'object')"

    if "claude" in model.lower():
        # Use AnthropicVertex for Claude models
        client = AnthropicVertex(region=region, project_id=project_id)
        
        # Convert prompt_input format to Anthropic format
        system_prompt = None
        messages = []
        for message in prompt_input:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_prompt = content if system_prompt is None else f"{system_prompt}\n\n{content}"
            elif role in ["user", "assistant"]:
                messages.append({
                    "role": role,
                    "content": content,
                })

        request_kwargs = {
            "max_tokens": max_completion_tokens,
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt

        response = client.messages.create(**request_kwargs)
        
        if return_type.lower() in ["string", "str", "text", "txt"]:
            return response.content[0].text.strip()
        elif return_type.lower() in ["json", "dict", "obj", "object", "response"]:
            return response
            
    elif "gemini" in model.lower():
        # Use Google GenAI for Gemini models
        client = genai.Client(
            vertexai=False, api_key=os.getenv("VERTEX_API_KEY")
        )
        
        # Convert prompt_input to Gemini format (just the content)
        contents = []
        for message in prompt_input:
            contents.append(message["content"])
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
        
        if return_type.lower() in ["string", "str", "text", "txt"]:
            return response.text.strip()
        elif return_type.lower() in ["json", "dict", "obj", "object", "response"]:
            return response
    
    else:
        raise ValueError(f"Unsupported model: {model}")


# Note: Vertex AI doesn't have batch processing like OpenAI
# For this use case, we'll use direct API calls
