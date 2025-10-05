# ollama_client.py — MinimalLLM-compatible Ollama client (streaming via /api/chat)
import os
import time
import json
from typing import Optional, Generator, List, Dict
import logger

try:
    import requests
except Exception as _e:
    raise RuntimeError(
        "The 'requests' package is required for ollama_client.py. "
        "Install with: pip install requests"
    ) from _e


class OllamaLLM:
    """
    MinimalLLM-compatible wrapper for a local Ollama server.
    - Uses /api/chat so we can preserve conversation history + system prompt.
    - Streams NDJSON lines and yields incremental assistant content.
    """

    def __init__(
        self,
        model: str = "type32/lemonade-rp:latest",
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant. Be concise and conversational.",
        keep_alive: str = "30m",
        temperature: float = 0.7,
        num_predict: int = 512,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)

        # Conversation history (Ollama-style)
        # We keep it OpenAI-like internally and translate on call.
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        logger.info(f"🧠 Initialized Ollama LLM: model='{self.model}' url='{self.base_url}'", module="llm_ollama")

    # ---------- helpers ----------
    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _abort_url(self) -> str:
        return f"{self.base_url}/api/abort"

    def _ollama_messages(self) -> List[Dict[str, str]]:
        """
        Convert our in-memory messages to the structure that Ollama expects.
        Ollama supports 'system' role in chat messages; we pass it through.
        """
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    # ---------- public API (MinimalLLM-compatible) ----------
    def prewarm(self):
        """Load the model into memory to avoid cold-start lag."""
        logger.info("Prewarming Ollama model...", module="llm_ollama")
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 8
                }
            }
            r = requests.post(self._chat_url(), json=payload, timeout=30)
            r.raise_for_status()
            logger.info("Ollama prewarm ok", module="llm_ollama")
        except Exception as e:
            logger.warn(f"Ollama prewarm failed (continuing anyway): {e}", module="llm_ollama")

    def stream_response(self, user_input: str) -> Generator[str, None, None]:
        """
        Stream assistant content as it’s generated.
        Yields small strings as Ollama produces them.
        """
        # Append user message
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": self._ollama_messages(),
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict
            }
        }

        start_time = time.time()
        first_token_time = None
        assistant_response = []

        logger.info(f"🧠 OLLAMA: Creating stream for: '{user_input[:30]}...'", module="llm_ollama")
        try:
            with requests.post(self._chat_url(), json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                logger.info("🧠 OLLAMA: Stream started…", module="llm_ollama")

                for raw in r.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        # NDJSON noise or partial line; skip
                        continue

                    # Typical frame: {"message":{"role":"assistant","content":"..."}, "done":false, ...}
                    msg = obj.get("message") or {}
                    chunk = msg.get("content", "")

                    if chunk:
                        if first_token_time is None:
                            first_token_time = time.time()
                            logger.debug(f"Ollama first token after {(first_token_time - start_time)*1000:.1f}ms",
                                      module="llm_ollama")
                        assistant_response.append(chunk)
                        yield chunk

                    if obj.get("done"):
                        break

            full = "".join(assistant_response).strip()
            logger.info(f"🧠 OLLAMA: Stream completed. Total response: '{full[:80]}{'...' if len(full)>80 else ''}'",
                     module="llm_ollama")

            # Add assistant message to history; trim history length
            if full:
                self.messages.append({"role": "assistant", "content": full})
                if len(self.messages) > 11:
                    self.messages = [self.messages[0]] + self.messages[-10:]

        except Exception as e:
            logger.error(f"❌ OLLAMA error: {e}", module="llm_ollama")
            yield f"Sorry, I encountered an error: {str(e)}"

    def generate_quick_response(self, user_input: str) -> str:
        """Non-streaming quick reply."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt + " Be very brief."},
                {"role": "user", "content": user_input},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": 64},
        }
        try:
            r = requests.post(self._chat_url(), json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            msg = (data.get("message") or {}).get("content", "")
            return msg or "I'm thinking..."
        except Exception as e:
            logger.error(f"🧠 OLLAMA quick response failed: {e}", module="llm_ollama")
            return "I'm thinking..."

    def abort_generation(self):
        """Abort any running generation (global to the server)."""
        try:
            requests.post(self._abort_url(), timeout=2)
            logger.info("Sent abort to Ollama server", module="llm_ollama")
        except Exception as e:
            logger.warn(f"Ollama abort failed (continuing): {e}", module="llm_ollama")

    def clear_history(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.info("Ollama conversation history cleared", module="llm_ollama")

    def measure_latency(self) -> float:
        """Round-trip latency for a tiny non-stream chat."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_predict": 4},
        }
        start = time.time()
        try:
            r = requests.post(self._chat_url(), json=payload, timeout=60)
            r.raise_for_status()
            ms = (time.time() - start) * 1000.0
            logger.info(f"Ollama latency: {ms:.1f}ms", module="llm_ollama")
            return ms
        except Exception as e:
            logger.warn(f"Ollama latency check failed: {e}", module="llm_ollama")
            return 0.0


if __name__ == "__main__":
    o = OllamaLLM()
    o.prewarm()
    o.measure_latency()
    print("Say hello (stream): ", end="", flush=True)
    for c in o.stream_response("Say hello in one short sentence."):
        print(c, end="", flush=True)
    print()
