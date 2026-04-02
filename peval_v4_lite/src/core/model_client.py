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
                # Use cached results if available
                if ModelClient._discovered_model:
                    self.client = ModelClient._cached_client
                    self.model = ModelClient._discovered_model
                    return

                self.client = openai.OpenAI(
                    base_url="https://api.openai.com/v1",
                    api_key=self.config.OPENAI_API_KEY, 
                    timeout=300
                )
                
                # Automated Precision Discovery
                try:
                    from .logger import PEVLogger
                    # Log discovered IDs and endpoint for debugging
                    PEVLogger.info(f"OpenAI Discovery Endpoint: {self.client.base_url}")
                    available_ids = [m.id for m in self.client.models.list()]
                    PEVLogger.info(f"Discovered Intelligence models: {available_ids if available_ids else 'None Found'}")
                    
                    # Prioritization Matrix (Using literal ID matches)
                    # Goliath Intelligence: Priority to GPT-5.4-Pro
                    candidates = ["gpt-5.4-pro", "gpt-5.4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                    
                    found_model = next((c for c in candidates if c in available_ids), None)
                    if not found_model and self.config.OPENAI_API_KEY:
                        # Intelligence Force-Push: Target GPT-4o directly if discovery fails
                        self.model = "gpt-4o"
                        PEVLogger.warn(f"Empty discovery list. Force-Pushing request to Intelligence Tier: {self.model}")
                    else:
                        self.model = found_model or "gpt-3.5-turbo"
                    
                    # Cache the discovery
                    ModelClient._discovered_model = self.model
                    ModelClient._cached_client = self.client
                    PEVLogger.success(f"Handshake Complete. Target: {self.model}")
                except Exception as e:
                    from .logger import PEVLogger
                    PEVLogger.warn(f"Model Discovery failed: {str(e)}. Falling back to Local Gaudi Model.")
                    self.model = self.config.USER_MODEL # Fallback to 8B/14B Gaudi simulator
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
            # Future-Proofing for OpenAI's new response endpoints
            if "gpt-5" in self.model:
                text_input = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages])
                response = self.client.responses.create(
                    model=self.model,
                    input=text_input
                )
                return response.output_text
            else:
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
