# backend/app/provider_client.py
from typing import List
import logging
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ProviderClient:
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement generate")


class NVIDIAProvider(ProviderClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")

        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # No local embeddings — ChromaDB uses built-in all-MiniLM-L6-v2
        logger.info("Using ChromaDB built-in embeddings (all-MiniLM-L6-v2)")

        self._test_connection()
        logger.info(f"NVIDIA NIM Provider initialized with model: {self.model}")

    def _test_connection(self):
        """Test connection to NVIDIA NIM API"""
        try:
            test_response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=10
            )
            if test_response.status_code == 200:
                logger.info("NVIDIA NIM API connection successful")
            else:
                logger.warning(f"NVIDIA API test returned: {test_response.status_code}")
        except Exception as e:
            logger.error(f"NVIDIA NIM connection test failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate response using NVIDIA NIM"""
        try:
            logger.info(f"Generating response via NVIDIA NIM ({self.model})...")

            messages = [
                {
                    "role": "system",
                    "content": """You are "Nyaya Mitra" - India's legal expert. 
                    Provide accurate legal information based strictly on the provided context. 
                    Be practical and helpful. 
                    Use relevant laws from context. 
                    When explaining procedures, give step-by-step instructions. 
                    Mention BNS/IPC when applicable.""",
                },
                {"role": "user", "content": prompt},
            ]

            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stream": False,
                    
                },
                timeout=60,
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()

                # Clean up whitespace
                import re
                generated_text = re.sub(r"\n\s*\n", "\n\n", generated_text)

                logger.info(f"NVIDIA NIM success: {len(generated_text)} chars in {elapsed_time:.2f}s")
                return generated_text

            elif response.status_code == 401:
                logger.error("NVIDIA API key invalid or expired")
                return self._get_auth_error_response()

            elif response.status_code == 429:
                logger.warning("NVIDIA NIM rate limit reached")
                return self._get_rate_limit_response()

            else:
                error_text = response.text[:200] if response.text else "No details"
                logger.error(f"NVIDIA NIM error {response.status_code}: {error_text}")
                return self._get_fallback_response()

        except requests.exceptions.Timeout:
            logger.error("NVIDIA NIM request timeout")
            return self._get_timeout_response()

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to NVIDIA NIM")
            return self._get_connection_error_response()

        except Exception as e:
            logger.error(f"NVIDIA NIM generation error: {e}")
            return self._get_fallback_response()

    def set_model(self, model_name: str):
        """Change the model (optional)"""
        supported_models = [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "google/gemma-2-9b-it",
        ]
        if model_name in supported_models:
            self.model = model_name
            logger.info(f"Changed NVIDIA model to: {model_name}")
        else:
            logger.warning(f"Model {model_name} not supported. Keeping: {self.model}")

    # Fallback responses (same as before)
    def _get_fallback_response(self) -> str:
        return """I couldn't access the legal analysis service at the moment. 

However, based on the legal documents in our database:

• Relevant legal sections have been retrieved
• The information is based on Indian laws (BNS, IPC, etc.)
• Please try again shortly for detailed analysis

For urgent legal help, consult a qualified professional."""

    def _get_auth_error_response(self) -> str:
        return """System Notice: AI Service Authentication Error

Authentication failed. Relevant documents retrieved, but detailed analysis unavailable.

Contact administrator to check NVIDIA API key."""

    def _get_rate_limit_response(self) -> str:
        return """Legal Analysis - Service Busy

High demand currently. Key sections retrieved from documents.

Please try again in a moment."""

    def _get_timeout_response(self) -> str:
        return """Legal Analysis - Processing

Service taking longer than expected. References retrieved.

Try again for full analysis."""

    def _get_connection_error_response(self) -> str:
        return """System Notice: AI Service Unavailable

Cannot connect to analysis service. Documents retrieved successfully.

Try again soon."""


def get_best_provider() -> ProviderClient:
    """Return the NVIDIA provider (only one used)"""
    logger.info("Initializing NVIDIA Provider...")

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if not nvidia_key:
        logger.error("NVIDIA_API_KEY not found!")
        raise ValueError(
            "NVIDIA_API_KEY required in .env file:\n"
            "NVIDIA_API_KEY=your_key_here"
        )

    try:
        provider = NVIDIAProvider()
        logger.info(f"NVIDIA NIM provider ready: {provider.model}")
        return provider
    except Exception as e:
        logger.error(f"NVIDIA init failed: {e}")
        raise Exception("NVIDIA provider failed. Check API key, internet, and NVIDIA status.")