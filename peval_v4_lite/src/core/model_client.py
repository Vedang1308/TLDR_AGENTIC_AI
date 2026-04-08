import openai
from typing import List, Optional, Dict
from .config import PEVConfig

class PEVInferenceError(Exception):
    """Custom error for model timeouts and connection failures."""
    pass

class ModelClient:
    """
    Shared synchronous interface to model endpoints.
    Ensures sequential execution to prevent VRAM over-saturation.
    """
    # Class-level cache to prevent 'Excess Info' and redundant handshaking
    _discovered_model = None
    _cached_client = None

    def __init__(self, mode: str = "agent"):
        self.config = PEVConfig()
        if mode == "agent":
            self.client = openai.OpenAI(base_url=self.config.AGENT_ENDPOINT, api_key="empty", timeout=300)
            self.model = self.config.AGENT_MODEL
        elif mode == "user":
            self.client = openai.OpenAI(base_url=self.config.USER_ENDPOINT, api_key="empty", timeout=300)
            self.model = self.config.USER_MODEL
        elif mode == "summarizer":
            if self.config.OPENAI_API_KEY:
                # PERFORMANCE PRIORITY: Always use OpenAI if key is present
                if ModelClient._discovered_model:
                    self.client = ModelClient._cached_client
                    self.model = ModelClient._discovered_model
                    return
                
                from .logger import PEVLogger
                PEVLogger.info("OpenAI Key Found. Prioritizing Intelligence Tier (GPT-4o) for all internal reasoning.")
                self.client = openai.OpenAI(
                    base_url="https://api.openai.com/v1",
                    api_key=self.config.OPENAI_API_KEY,
                    timeout=300
                )
                
                # Discovery Handshake
                try:
                    available_ids = [m.id for m in self.client.models.list()]
                    candidates = ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
                    found_model = next((c for c in candidates if c in available_ids), None)
                    self.model = found_model or "gpt-4o"
                    ModelClient._discovered_model = self.model
                    ModelClient._cached_client = self.client
                    PEVLogger.success(f"Handshake Complete. Target: {self.model}")
                except Exception as e:
                    PEVLogger.warn(f"Discovery Failed: {e}. Defaulting to gpt-4o.")
                    self.model = "gpt-4o"
                return

            # UNIFIED AGENT MODEL: If no OpenAI key, use the primary Agent model for EVERYTHING.
            # This removes the 'Third LLM' (72B Summarizer) complexity.
            from .logger import PEVLogger
            PEVLogger.info(f"Unified Intelligence: Using primary Agent model ({self.config.AGENT_MODEL}) for internal checks & summarization.")
            self.client = openai.OpenAI(
                base_url=self.config.AGENT_ENDPOINT,
                api_key="empty",
                timeout=300
            )
            self.model = self.config.AGENT_MODEL

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Simple string-in, string-out wrapper for backward compatibility."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            from .logger import PEVLogger
            PEVLogger.error(f"Generate failure: {str(e)}")
            raise PEVInferenceError(str(e))

    def chat(self, messages: List, temperature: float = 1.0):
        try:
            # Unified Execution: Standard completions API only
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except openai.NotFoundError as e:
            from .logger import PEVLogger
            PEVLogger.warn(f"Model ID '{self.model}' not found in your OpenAI account. Triggering Resilience Fallback...")
            
            # Resilience Chain logic
            if "gpt-5" in self.model:
                PEVLogger.info("Falling back to gpt-4o...")
                self.model = "gpt-4o"
                return self.chat(messages, temperature)
            elif self.model == "gpt-4o":
                PEVLogger.info("Falling back to gpt-4o-mini...")
                self.model = "gpt-4o-mini"
                return self.chat(messages, temperature)
            elif self.model == "gpt-4o-mini":
                PEVLogger.info("Falling back to gpt-3.5-turbo...")
                self.model = "gpt-3.5-turbo"
                return self.chat(messages, temperature)
            elif self.model == "gpt-3.5-turbo":
                PEVLogger.warn("OpenAI API inaccessible. Final fallback: Local 8B Gaudi Model.")
                self.client = openai.OpenAI(base_url=self.config.USER_ENDPOINT, api_key="empty", timeout=300)
                self.model = self.config.USER_MODEL
                return self.chat(messages, temperature)
            else:
                PEVLogger.error(f"Intelligence failure: {str(e)}")
                raise PEVInferenceError(str(e))
        except Exception as e:
            from .logger import PEVLogger
            PEVLogger.error(f"Inference failure: {str(e)}")
            raise PEVInferenceError(str(e))
