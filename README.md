Chatty — low-latency voice assistant pipeline
=============================================

Small, minimal voice chat pipeline that glues realtime STT -> LLM -> streaming TTS together.
This repository wires a local or cloud LLM (Ollama / OpenAI adapters) to a streaming TTS engine (RealtimeTTS / Kokoro) and a lightweight audio player so you can have conversational voice interactions with low latency.

Highlights
- simple single-file pipeline orchestration: `pipeline.py`
- pluggable LLM adapters: `llm_ollama_adapter.py`, `llm_openai_adapter.py`
- shared adapter helpers in `llm_interface.py`
- realtime TTS via `RealtimeTTS` Kokoro engine (configured in `pipeline.py`)
- lightweight metrics: per-turn QUESTION-TO-RESPONSE_MS and LLM timing (TTF / total)

Quick start
-----------
Prerequisites
- Python 3.10/3.11 recommended (see `requirements.txt`)
- A virtualenv dedicated to this project
- If you use Kokoro (recommended for high-quality local TTS) you will need a suitable CUDA-enabled GPU or CPU fallback depending on models used.

Install (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Run

```powershell
python pipeline.py
```

Configuration
-------------
- `config.json` is the single source of user-facing configuration: it contains the LLM backend choice, `system_prompt`, and per-backend `opts` (model names, `api_key_env`, moderation settings, etc.). Do not edit `pipeline.py` for configuration changes.

Key files
- `pipeline.py` — orchestrates STT -> LLM -> TTS loop and exposes per-turn metrics
- `llm_interface.py` — shared helpers and adapter protocol
- `llm_ollama_adapter.py` / `llm_openai_adapter.py` — concrete adapters for Ollama and OpenAI
- `RealtimeTTS` integration lives under the system venv (Kokoro engine used by default)

Metrics
-------
At the end of each turn the pipeline logs three timing metrics (ms):

- QUESTION-TO-RESPONSE_MS — wall-clock from end of user speech (when the question is submitted) to first audio played by the player
- LLM-TTF_MS — LLM time-to-first-token (how long the LLM took to return the first token)
- LLM-TOTAL_MS — total time spent generating the LLM response

These metrics let you separate LLM latency from TTS latency and evaluate where to optimize.

Kokoro / custom voices
-----------------------
This project uses RealtimeTTS's Kokoro engine by default. Kokoro loads voice tensors via its `KPipeline`. If you want to use your own `.pth` voice file you have two options:

1. If your voice package is a directory matching KPipeline expectations you can point KPipeline at a local path (replace `repo_id` with a filesystem path in `RealtimeTTS` / `KokoroEngine` initialization).
2. If you have a raw voice `.pth` tensor, you may register it into the `KokoroEngine.blended_voices` cache and call `engine.set_voice("my_custom_voice")` (the code must match the tensor shape Kokoro expects).

See `RealtimeTTS` docs for details. Be careful with model sizes — large models require GPU and may increase startup time.
-
Producer / consumer TTS design (threads)
---------------------------------------

This project uses a producer/consumer pattern for TTS to minimize audible latency and allow smooth playback while synthesis continues. In our pipeline:

- The producer thread (synthesizer) converts text into small "synthesis blocks" / audio chunks and enqueues them as they become available.
- The consumer thread (audio player) dequeues ready blocks and plays them immediately. Playback runs independently of synthesis, so the player can start as soon as the first chunk is ready.

Why this helps
- Starts audio playback earlier (reduces perceived latency). The player doesn't wait for the entire response.
- Allows synthesis to run in parallel and keep the audio buffer topped up.
- Makes interruption simpler: the player can stop consumption immediately while the synthesizer can be signalled to cancel or drain.

Implementation notes and recommendations
- Use a thread-safe queue (e.g., `queue.Queue`) for communication between the synthesizer and player.
- Keep synthesis blocks moderately small (e.g., 100–500ms of audio) to balance latency and overhead.
- Provide a clear first-play callback (used in `pipeline.py`) to measure QUESTION-TO-RESPONSE_MS — set the question end timestamp before sending to the LLM and record the time when the player plays the first block.
- Support interrupt/stop signals: when the user interrupts, stop the player immediately and signal the synthesizer to cancel (or drain and stop producing further blocks).
- Consider a small buffer target (e.g., 2–4 blocks) to avoid underflow while keeping memory use small.

Small conceptual example
```python
from queue import Queue
from threading import Thread, Event

blocks = Queue()
stop_event = Event()

def synthesizer(text):
	for chunk in synth_stream(text):               # yields audio blocks
		if stop_event.is_set(): break
		blocks.put(chunk)
	blocks.put(None)  # sentinel: synthesis finished

def player():
	while True:
		blk = blocks.get()
		if blk is None: break
		play_block(blk)

Thread(target=player, daemon=True).start()
Thread(target=lambda: synthesizer("Hello world"), daemon=True).start()
```

Notes
- The concrete `OptimizedKokoroTTS` in `pipeline.py` implements this approach and records `question_end_time` and `last_q_to_response_ms` on first playback for the metrics above.
- Tune block size and queue depth for your hardware and network conditions. Smaller blocks reduce first-play latency but increase processing overhead.

Licensing and attribution
-------------------------
This project is released under the MIT License (see `LICENSE`). It depends on third-party packages which have their own licenses. Most core TTS infrastructure here (`RealtimeTTS`) is MIT-licensed; kokoro model weights / checkpoints may have different licensing terms — verify before redistributing any third-party model files.

Security & privacy
------------------
- Local-only model files (Ollama / Kokoro) keep data on-device. If you use cloud backends (OpenAI, Azure), be aware that user text and prompts are sent to those services.

Contributing
------------
Contributions are welcome. Please open issues or PRs. If you add support for additional backends or engines, add config examples and small smoke-tests.

Contact / credits
-----------------
Built with RealtimeTTS (https://github.com/KoljaB/RealtimeTTS) and local LLM adapters.

