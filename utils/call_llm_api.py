"""
LLM API interface module.
Handles communication with LLM providers (OpenAI, Azure OpenAI, etc.).
"""

import os
import time
import json
import requests
import re
from typing import Optional

from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()

class LLMCompletionCall:
    """
    A class to handle interactions with LLM APIs using the OpenAI client interface.
    Supports standard OpenAI API and Azure OpenAI.
    """

    def __init__(self):
        """
        Initialize the LLMCompletionCall instance.

        Loads configuration from environment variables:
        - LLM_MODEL: The model name to use (default: "deepseek-chat").
        - LLM_BASE_URL: The base URL for the API (default: "https://api.deepseek.com").
        - LLM_API_KEY: The API key (required).
        - OPENAI_PROVIDER: The provider type, "openai" or "azure" (default: "openai").
        - API_VERSION: API version for Azure OpenAI (default: "2025-01-01-preview").

        Raises:
            ValueError: If LLM_API_KEY is not provided.
        """
        self.llm_model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        if not self.llm_api_key:
            raise ValueError("LLM API key not provided")
        self.openai_provider = os.getenv("OPENAI_PROVIDER", "openai").lower()
        if self.openai_provider == "azure":
            self.api_version = os.getenv("API_VERSION", "2025-01-01-preview")
            self.client = AzureOpenAI(
                    azure_endpoint=self.llm_base_url,
                    api_key=self.llm_api_key,
                    api_version=self.api_version,
                )
        else:
            self.client = OpenAI(base_url=self.llm_base_url, api_key = self.llm_api_key)

    def call_api(self, content: str) -> str:
        """
        Call the LLM API to generate text based on the provided content.
        
        Args:
            content (str): The prompt content to send to the LLM.
            
        Returns:
            str: The generated text response from the LLM, cleaned of markdown code fences if present.

        Raises:
            Exception: If the API call fails.
        """
            
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": content}],
                temperature=0.3
            )
            raw = completion.choices[0].message.content or ""
            clean_completion = self._clean_llm_content(raw)
            return clean_completion
            
        except Exception as e:
            logger.error(f"LLM api calling failed. Error: {e}")
            raise e 

    def _clean_llm_content(self, text: str) -> str:
        """
        Clean the raw content returned by the LLM.

        Removes markdown code fences (```json ... ```) and extra whitespace.

        Args:
            text (str): The raw text from the LLM.

        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        m = fence_re.match(t)
        if m:
            t = m.group("body").strip()
        else:
            if t.startswith("```") and t.endswith("```") and len(t) >= 6:
                t = t[3:-3].strip()

        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()

        return t
