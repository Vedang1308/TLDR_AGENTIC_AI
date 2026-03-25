import openai
from .config import PEVConfig

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
                api_key="empty"
            )
            self.model = self.config.AGENT_MODEL
        elif mode == "user":
            self.client = openai.OpenAI(
                base_url=self.config.USER_ENDPOINT,
                api_key="empty"
            )
            self.model = self.config.USER_MODEL
        elif mode == "summarizer":
            self.client = openai.OpenAI(
                api_key=self.config.OPENAI_API_KEY
            )
            self.model = self.config.SUMMARIZER_MODEL

    def chat(self, messages: list, temperature: float = 0.0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
