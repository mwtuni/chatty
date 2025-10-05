from typing import Protocol, Generator, List, Callable, Iterable, Any, Optional
import logger


class BaseLLMAdapter:
    """Small concrete base class with shared helpers used by adapters.

    It implements common behaviors that were duplicated across adapters:
    - composing a history text payload
    - composing the final payload to send to the client
    - buffering streaming chunks into words and appending to history
    - lightweight init/validation and prewarm helpers

    Adapters should still construct the concrete client instance themselves
    (OpenAI/Ollama specifics stay in the client modules), but can call these
    helpers to reduce duplication.
    """

    history: List[tuple[str, str]] = []
    history_max: int = 10

    def compose_history_text(self) -> str:
        if not getattr(self, "history", None):
            return ""
        return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in self.history)

    def compose_payload(self, question: str) -> str:
        history_text = self.compose_history_text()
        if history_text:
            return f"{history_text}\n\n{question}"
        return question

    def stream_words(self, response_stream, question: str, logger_module: str = "llm") -> Generator[str, None, None]:
        """Consume a low-level response stream and yield words.

        This mirrors the buffering logic used previously in both adapters and
        also appends the final assistant text to `self.history`.
        """
        assistant_accum = ""
        word_buffer = ""
        total_words = 0

        for chunk in response_stream:
            word_buffer += chunk
            assistant_accum += chunk

            words = word_buffer.split()
            if len(words) > 1:
                for word in words[:-1]:
                    total_words += 1
                    try:
                        logger.debug(f"🧠 [{total_words:3d}] '{word}'", module=logger_module)
                    except Exception:
                        pass
                    yield word
                word_buffer = words[-1]

        if word_buffer.strip():
            total_words += 1
            try:
                logger.debug(f"🧠 [{total_words:3d}] '{word_buffer.strip()}' (final)", module=logger_module)
            except Exception:
                pass
            yield word_buffer.strip()

        # Save assistant response into runtime history (trim to history_max)
        try:
            if assistant_accum.strip():
                # ensure history exists
                if not getattr(self, "history", None):
                    self.history = []
                self.history.append((question, assistant_accum.strip()))
                if len(self.history) > int(getattr(self, "history_max", 10)):
                    self.history = self.history[-int(getattr(self, "history_max", 10)):]
        except Exception:
            pass

        try:
            logger.info(f"🧠 LLM response complete: {total_words} words", module=logger_module)
        except Exception:
            pass

    # --- New shared helpers -------------------------------------------------
    def validate_and_set_config(
        self,
        model: Optional[str],
        system_prompt: Optional[str],
        *,
        history_max: Optional[int] = None,
        backend_name: str = "LLM",
    ) -> None:
        """Validate required config values and set instance attributes.

        Raises ValueError with a clear message if required values are missing.
        """
        if not system_prompt or not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(f"{backend_name} requires 'system_prompt' from config.json (llm.system_prompt).")
        if not model or not isinstance(model, str) or not model.strip():
            raise ValueError(f"{backend_name} requires a model name. Please set the backend 'opts.model' in config.json.")

        # set runtime attributes
        self.system_prompt = system_prompt
        self.history = []
        if history_max is not None:
            try:
                self.history_max = int(history_max)
            except Exception:
                self.history_max = 10

    def safe_log_system_prompt(self, max_chars: int = 1024, module: str = "llm") -> None:
        """Log the system prompt defensively (truncate and swallow logging errors)."""
        try:
            sp = getattr(self, "system_prompt", "") or ""
            if len(sp) > max_chars:
                sp = sp[:max_chars] + "\n...(truncated)"
            logger.info("System prompt:\n" + sp, module=module)
        except Exception:
            pass

    def prewarm_client(self, client: Any, backend_name: str = "LLM", module: str = "llm") -> None:
        """Call client.prewarm() with consistent logging and error handling."""
        try:
            logger.info(f"🔥 Prewarming {backend_name} model...", module=module)
            client.prewarm()
            logger.info(f"✅ {backend_name} model prewarmed", module=module)
        except Exception as e:
            logger.error(f"❌ Failed to prewarm {backend_name} model: {e}", module=module)
            raise

    def ask_common_wrapper(
        self,
        client_stream_fn: Callable[[Any], Iterable[str]],
        question: str,
        *,
        prepend_system: bool = False,
        logger_module: str = "llm",
    ) -> Generator[str, None, None]:
        """Common orchestration used by adapters to ask the LLM and stream words.

        client_stream_fn: a callable that accepts a single payload argument and
            returns an iterable/iterator of string chunks.
        prepend_system: if True, the adapter's system_prompt will be prepended to
            the composed payload text (defensive guard for backends that may
            not honor structured system roles).
        """
        logger.info(f"🧠 Asking LLM: '{question}'", module=logger_module)
        try:
            # reset per-request LLM metrics
            try:
                self.last_llm_ttf_ms = None
                self.last_llm_total_ms = None
            except Exception:
                pass

            payload = self.compose_payload(question)
            if prepend_system and getattr(self, "system_prompt", None):
                try:
                    payload = f"{self.system_prompt}\n\n{payload}"
                except Exception:
                    pass

            import time
            start = time.monotonic()
            response_stream = client_stream_fn(payload)

            # capture time to first token
            first = True
            def _iter():
                nonlocal first
                for chunk in response_stream:
                    if first:
                        try:
                            ttf = time.monotonic()-start
                            # persist metrics in milliseconds for external reporting
                            try: self.last_llm_ttf_ms = ttf * 1000.0
                            except Exception: pass
                            logger.info(f"LLM-TTF: {ttf:.3f}s", module=logger_module)
                        except Exception:
                            pass
                        first = False
                    yield chunk

            for w in self.stream_words(_iter(), question, logger_module=logger_module):
                yield w

            try:
                total = time.monotonic()-start
                try: self.last_llm_total_ms = total * 1000.0
                except Exception: pass
                logger.info(f"LLM-TOTAL: {total:.3f}s", module=logger_module)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"❌ LLM error: {e}", module=logger_module)
            for w in f"Sorry, I encountered an error: {e}".split():
                yield w

    # ----------------------------------------------------------------------


class LLMAdapter(Protocol):
    """Structural protocol describing the LLM adapter public surface.

    Implementations should provide a streaming interface that yields words
    (or small chunks) for TTS to consume.
    """

    history: List[tuple[str, str]]

    def ask_streaming(self, question: str) -> Generator[str, None, None]: ...

    def ask_question(self, question: str) -> Generator[str, None, None]: ...

    def clear_history(self) -> None: ...

    # Optional hooks
    def prewarm(self) -> None: ...
    def abort_generation(self) -> None: ...
