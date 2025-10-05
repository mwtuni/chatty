# llm_openai_adapter.py - OpenAI LLM adapter
"""🧠 Simple LLM streaming interface (OpenAI backend)"""
import os
from typing import Generator
from openai_client import MinimalLLM as _OpenAIClient
from llm_interface import LLMAdapter, BaseLLMAdapter
import logger

class OpenAILLM(BaseLLMAdapter):
    """OpenAI backend implementation of the LLMAdapter protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        history_max: int = 5,
    ):
        # Validate and set common config attributes
        self.validate_and_set_config(model, system_prompt, history_max=history_max, backend_name="OpenAI")

        try:
            self.llm = _OpenAIClient(
                api_key=api_key,
                model=model,
                system_prompt=self.system_prompt,
            )
            logger.info(f"🧠 OpenAILLM initialized with {model}", module="llm")
            # Log the system prompt for visibility (helpful when diagnosing persona issues)
            self.safe_log_system_prompt(module="llm")

            # Prewarm the LLM
            self.prewarm_client(self.llm, backend_name="OpenAI", module="llm")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}", module="llm")
            raise
    
    def ask_streaming(self, question: str) -> Generator[str, None, None]:
        yield from self.ask_common_wrapper(self.llm.stream_response, question, prepend_system=True, logger_module="llm")
       
    def ask_question(self, question: str) -> Generator[str, None, None]:
        return self.ask_streaming(question)

    def clear_history(self):
        """Clear the runtime conversation history."""
        self.history.clear()
