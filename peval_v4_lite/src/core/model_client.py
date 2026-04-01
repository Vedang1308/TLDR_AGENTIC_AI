import openai
from typing import List
from .config import PEVConfig

class PEVInferenceError(Exception):
    """Custom error for model timeouts and connection failures."""
    pass

class ModelClient:
    """
    Shared synchronous interface to model endpoints.
    Ensures sequential execution to prevent VRAM over-saturation.
    """
    def __init__(self, mode: str = "agent"):
        self.config = PEVConfig()
        if mode == "agent":
            self.client = openai.OpenAI(
                base_url=self.config.AGENT_ENDPOINT,
                api_key="empty",
                timeout=300
            )
            self.model = self.config.AGENT_MODEL
        elif mode == "user":
            self.client = openai.OpenAI(
                base_url=self.config.USER_ENDPOINT,
                api_key="empty",
                timeout=300
            )
            # Fix: Ensure USER_MODEL is used
            self.model = self.config.USER_MODEL
        elif mode == "summarizer":
            if self.config.OPENAI_API_KEY:
                self.client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY, timeout=300)
                
                # Automated Model Discovery: Find the best GPT-4 model available
                try:
                    from .logger import PEVLogger
                    PEVLogger.info("Discovering available GPT-4 models...")
                    available_models = [m.id for m in self.client.models.list()]
                    
                    # Prioritization Matrix
                    candidates = ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"]
                    self.model = next((c for c in candidates if c in available_models), "gpt-3.5-turbo")
                    
                    PEVLogger.success(f"Handshake Complete. Using Intelligence Model: {self.model}")
                except Exception as e:
                    from .logger import PEVLogger
                    PEVLogger.warn(f"Model Discovery failed: {str(e)}. Falling back to default.")
                    self.model = self.config.SUMMARIZER_MODEL
            elif self.config.OPENROUTER_API_KEY:
                self.client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.config.OPENROUTER_API_KEY,
                    timeout=300
                )
                self.model = self.config.OPENROUTER_MODEL
            else:
                # Fallback to local 8B User Simulator for higher quality summarization 
                # (instead of the 4B Agent)
                from .logger import PEVLogger
                PEVLogger.warn("No API key found. Falling back to local 8B User vLLM for summarization.")
                self.client = openai.OpenAI(
                    base_url=self.config.USER_ENDPOINT,
                    api_key="empty",
                    timeout=300
                )
                self.model = self.config.USER_MODEL

    def chat(self, messages: List, temperature: float = 1.0):
        try:
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
            if self.model == "gpt-4o-mini":
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
