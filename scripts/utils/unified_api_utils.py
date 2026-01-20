"""
Unified API utility that supports multiple providers (OpenAI, Anthropic, Vertex, OpenRouter).
Automatically selects the appropriate provider based on config settings.
"""

import os
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv

load_dotenv()


# Provider constants
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_VERTEX = "vertex"
PROVIDER_OPENROUTER = "openrouter"

SUPPORTED_PROVIDERS = [PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_VERTEX, PROVIDER_OPENROUTER]


def get_provider_from_config(config: dict) -> str:
    """
    Determines the API provider from config.
    
    Supports two config formats:
    1. New unified format: config['api']['provider']
    2. Legacy format: checks for 'openai', 'anthropic', 'vertex', 'openrouter' keys
    
    Returns:
        Provider name string
    """
    # Check for new unified format
    if 'api' in config and 'provider' in config['api']:
        provider = config['api']['provider'].lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {SUPPORTED_PROVIDERS}")
        return provider
    
    # Check for legacy format (provider-specific keys)
    for provider in SUPPORTED_PROVIDERS:
        if provider in config:
            return provider
    
    raise ValueError(f"No valid API provider found in config. Supported: {SUPPORTED_PROVIDERS}")


def get_api_settings(config: dict) -> dict:
    """
    Extracts API settings from config, supporting both unified and legacy formats.
    
    Returns:
        Dictionary with model, temperature, max_tokens, and provider-specific settings
    """
    provider = get_provider_from_config(config)
    
    # New unified format
    if 'api' in config and 'provider' in config['api']:
        api_config = config['api']
        settings = {
            'provider': provider,
            'model': api_config.get('model'),
            'temperature': api_config.get('temperature', 1.0),
            'max_tokens': api_config.get('max_tokens', 4096),
            'top_p': api_config.get('top_p', 1.0),
        }
        # Provider-specific settings
        if provider == PROVIDER_VERTEX:
            settings['region'] = api_config.get('region', 'us-east5')
            settings['project_id'] = api_config.get('project_id', '')
        if provider == PROVIDER_OPENROUTER:
            settings['reasoning_effort'] = api_config.get('reasoning_effort')
        if provider == PROVIDER_OPENAI:
            settings['reasoning_effort'] = api_config.get('reasoning_effort')
        return settings
    
    # Legacy format
    provider_config = config[provider]
    settings = {
        'provider': provider,
        'model': provider_config.get('model'),
        'temperature': provider_config.get('temperature', 1.0),
        'max_tokens': provider_config.get('max_tokens', 4096),
        'top_p': provider_config.get('top_p', 1.0),
    }
    # Provider-specific settings
    if provider == PROVIDER_VERTEX:
        settings['region'] = provider_config.get('region', 'us-east5')
        settings['project_id'] = provider_config.get('project_id', '')
    if provider == PROVIDER_OPENROUTER:
        settings['reasoning_effort'] = provider_config.get('reasoning_effort')
    if provider == PROVIDER_OPENAI:
        settings['reasoning_effort'] = provider_config.get('reasoning_effort')
    
    return settings


def create_response_chat(
    config: dict,
    prompt_input: List[Dict],
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    return_type: str = "string",
    **kwargs
) -> Any:
    """
    Unified chat response function that automatically selects the appropriate provider.
    
    Args:
        config: Configuration dictionary containing API settings
        prompt_input: Message list [{"role": "user/system/assistant", "content": "..."}]
        max_completion_tokens: Override max tokens (uses config value if not provided)
        temperature: Override temperature (uses config value if not provided)
        return_type: "string" for text response, "response" for full response object
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Response text if return_type is "string", full response object otherwise
    """
    settings = get_api_settings(config)
    provider = settings['provider']
    model = settings['model']
    
    # Use provided values or fall back to config values
    max_tokens = max_completion_tokens or settings['max_tokens']
    temp = temperature if temperature is not None else settings['temperature']
    
    if provider == PROVIDER_ANTHROPIC:
        return _call_anthropic(
            model=model,
            prompt_input=prompt_input,
            max_completion_tokens=max_tokens,
            temperature=temp,
            return_type=return_type
        )
    
    elif provider == PROVIDER_OPENAI:
        return _call_openai(
            model=model,
            prompt_input=prompt_input,
            max_completion_tokens=max_tokens,
            temperature=temp,
            top_p=settings.get('top_p', 1.0),
            return_type=return_type,
            reasoning_effort=settings.get('reasoning_effort') or kwargs.get('reasoning_effort')
        )
    
    elif provider == PROVIDER_VERTEX:
        return _call_vertex(
            model=model,
            prompt_input=prompt_input,
            max_completion_tokens=max_tokens,
            temperature=temp,
            region=settings.get('region', 'us-east5'),
            project_id=settings.get('project_id', ''),
            return_type=return_type
        )
    
    elif provider == PROVIDER_OPENROUTER:
        return _call_openrouter(
            model=model,
            prompt_input=prompt_input,
            max_completion_tokens=max_tokens,
            temperature=temp,
            top_p=settings.get('top_p', 1.0),
            return_type=return_type,
            reasoning_effort=settings.get('reasoning_effort') or kwargs.get('reasoning_effort')
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_response_chat_for_structured_output(
    config: dict,
    prompt_input: List[Dict],
    response_schema: Dict,
    max_completion_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    return_type: str = "string",
    **kwargs
) -> Any:
    """
    Unified structured output function (currently only supports OpenAI).
    
    Args:
        config: Configuration dictionary containing API settings
        prompt_input: Message list
        response_schema: JSON schema for structured output
        max_completion_tokens: Override max tokens
        temperature: Override temperature
        return_type: "string" for text response, "response"/"dict" for full response
        **kwargs: Additional arguments
    
    Returns:
        Response based on return_type
    """
    settings = get_api_settings(config)
    provider = settings['provider']
    
    if provider != PROVIDER_OPENAI:
        raise ValueError(f"Structured output is currently only supported for OpenAI provider, got: {provider}")
    
    from .openai_api_utils import create_response_chat_for_structured_output_of_reasoning_model
    
    max_tokens = max_completion_tokens or settings['max_tokens']
    temp = temperature if temperature is not None else settings['temperature']
    
    return create_response_chat_for_structured_output_of_reasoning_model(
        model=settings['model'],
        prompt_input=prompt_input,
        response_schema=response_schema,
        max_completion_tokens=max_tokens,
        temperature=temp,
        top_p=settings.get('top_p', 1.0),
        return_type=return_type,
        reasoning_effort=settings.get('reasoning_effort') or kwargs.get('reasoning_effort')
    )


# ============================================================================
# Provider-specific implementations (internal use)
# ============================================================================

def _call_anthropic(
    model: str,
    prompt_input: List[Dict],
    max_completion_tokens: int,
    temperature: float,
    return_type: str
) -> Any:
    """Internal: Call Anthropic API"""
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Separate system and user messages
    system_message = ""
    messages = []
    
    for msg in prompt_input:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            messages.append(msg)
    
    response = client.messages.create(
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        system=system_message,
        messages=messages
    )
    
    if return_type.lower() in ["string", "str", "text", "txt"]:
        return response.content[0].text.strip()
    else:
        return response


def _call_openai(
    model: str,
    prompt_input: List[Dict],
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    return_type: str,
    reasoning_effort: Optional[str] = None
) -> Any:
    """Internal: Call OpenAI API"""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    
    request_kwargs = {
        "model": model,
        "messages": prompt_input,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if reasoning_effort is not None:
        request_kwargs["reasoning_effort"] = reasoning_effort
    
    response = client.chat.completions.create(**request_kwargs)
    
    if return_type.lower() in ["string", "str", "text", "txt"]:
        return response.choices[0].message.content.strip()
    else:
        return response


def _call_vertex(
    model: str,
    prompt_input: List[Dict],
    max_completion_tokens: int,
    temperature: float,
    region: str,
    project_id: str,
    return_type: str
) -> Any:
    """Internal: Call Vertex AI API (Claude or Gemini)"""
    
    if "claude" in model.lower():
        from anthropic import AnthropicVertex
        
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
        else:
            return response
            
    elif "gemini" in model.lower():
        from google import genai
        
        client = genai.Client(
            vertexai=False, api_key=os.getenv("VERTEX_API_KEY")
        )
        
        # Convert prompt_input to Gemini format
        contents = []
        for message in prompt_input:
            contents.append(message["content"])
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
        )
        
        if return_type.lower() in ["string", "str", "text", "txt"]:
            return response.text.strip()
        else:
            return response
    
    else:
        raise ValueError(f"Unsupported Vertex model: {model}. Must contain 'claude' or 'gemini'")


def _call_openrouter(
    model: str,
    prompt_input: List[Dict],
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    return_type: str,
    reasoning_effort: Optional[str] = None
) -> Any:
    """Internal: Call OpenRouter API"""
    import requests
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": prompt_input,
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
    
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    response_data = response.json()
    
    if return_type.lower() in ["string", "str", "text", "txt"]:
        return response_data["choices"][0]["message"]["content"].strip()
    else:
        return response_data


# ============================================================================
# Utility functions
# ============================================================================

def validate_config(config: dict) -> bool:
    """
    Validates that the config has required API settings.
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        settings = get_api_settings(config)
        if not settings.get('model'):
            raise ValueError("Model must be specified in config")
        return True
    except Exception as e:
        raise ValueError(f"Invalid config: {str(e)}")


def get_required_env_var(provider: str) -> str:
    """
    Returns the required environment variable name for a provider.
    """
    env_vars = {
        PROVIDER_OPENAI: "OPENAI_API_KEY",
        PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
        PROVIDER_VERTEX: "GOOGLE_APPLICATION_CREDENTIALS or VERTEX_API_KEY",
        PROVIDER_OPENROUTER: "OPENROUTER_API_KEY"
    }
    return env_vars.get(provider, "Unknown")


def check_api_key(config: dict) -> bool:
    """
    Checks if the required API key is set for the configured provider.
    
    Returns:
        True if API key is set, False otherwise
    """
    provider = get_provider_from_config(config)
    
    if provider == PROVIDER_OPENAI:
        return bool(os.getenv("OPENAI_API_KEY"))
    elif provider == PROVIDER_ANTHROPIC:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    elif provider == PROVIDER_VERTEX:
        # Vertex can use either Google credentials or API key
        return bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_API_KEY"))
    elif provider == PROVIDER_OPENROUTER:
        return bool(os.getenv("OPENROUTER_API_KEY"))
    
    return False
