import os
import json
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_response_chat(
        self,
        model: str,
        prompt_input: List[Dict],
        max_completion_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        return_type: str = "string",
        reasoning_effort: Optional[str] = None
    ):
        """
        Generate chat response using OpenRouter API
        """
        assert return_type in ["string", "str", "text", "txt", "json", "dict", "obj", "object", "response"], \
            "return_type must be one of 'string' ('str', 'text', 'txt') or 'response' ('json', 'dict', 'obj', 'object')"
        
        payload = {
            "model": model,
            "messages": prompt_input,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            if return_type.lower() in ["string", "str", "text", "txt"]:
                return response_data["choices"][0]["message"]["content"].strip()
            elif return_type.lower() in ["json", "dict", "obj", "object", "response"]:
                return response_data
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from OpenRouter API: {str(e)}")


def create_openrouter_response(
    model: str,
    prompt_input: List[Dict],
    max_completion_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 1.0,
    return_type: str = "string",
    reasoning_effort: Optional[str] = None
):
    """
    Convenience function for generating responses using OpenRouter API
    """
    client = OpenRouterClient()
    return client.create_response_chat(
        model=model,
        prompt_input=prompt_input,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        return_type=return_type,
        reasoning_effort=reasoning_effort
    )
