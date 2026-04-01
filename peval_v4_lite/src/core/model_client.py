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
        except Exception as e:
            from .logger import PEVLogger
            PEVLogger.error(f"Inference failure: {str(e)}")
            raise PEVInferenceError(str(e))
