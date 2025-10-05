# llm_ollama_adapter.py — Ollama LLM adapter
"""🧠 Simple LLM streaming interface (Ollama backend)"""
import os
from typing import Generator
from ollama_client import OllamaLLM as _OllamaClient
from llm_interface import LLMAdapter, BaseLLMAdapter
import logger

class OllamaLLM(BaseLLMAdapter):
    """Ollama backend implementation of the LLMAdapter protocol."""

    def __init__(
    self,
    model: str | None = None,
    system_prompt: str | None = None,
        keep_alive: str = "360m",
        api_key: str | None = None,
        history_max: int = 10,
    ):
        # Validate config and set common instance attributes
        self.validate_and_set_config(model, system_prompt, history_max=history_max, backend_name="Ollama")

        try:
            # Instantiate client with structured system_prompt support
            self.llm = _OllamaClient(
                model=model,
                system_prompt=self.system_prompt,
                keep_alive=keep_alive,
            )
            logger.info(f"🧠 OllamaLLM initialized with {model}", module="llm")
            self.safe_log_system_prompt(module="llm")

            # Prewarm the local model (keeps it in VRAM/RAM)
            self.prewarm_client(self.llm, backend_name="Ollama", module="llm")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama LLM: {e}", module="llm")
            raise

    def ask_streaming(self, question: str) -> Generator[str, None, None]:
        yield from self.ask_common_wrapper(self.llm.stream_response, question, prepend_system=False, logger_module="llm")

    def ask_question(self, question: str) -> Generator[str, None, None]:
        return self.ask_streaming(question)

    def clear_history(self) -> None:
        """Clear the runtime conversation history."""
        self.history = []
