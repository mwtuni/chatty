# pipeline.py — voice loop with hard-cut + true synth cancel (lean)
import os, time, threading, logger, json
from queue import Queue
from mic_in_interrupt import stream_with_barge_in

# Resolve LLM backend with the following precedence:
# 1) config.json llm.backend (if present)
# 2) LLM_BACKEND environment variable
# 3) default 'ollama'
BACKEND = "ollama"
try:
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                be = cfg.get('llm', {}).get('backend')
                if isinstance(be, str) and be.strip():
                    BACKEND = be.strip().lower()
        except Exception:
            # ignore config parse errors and fall back to env/default
            pass
    else:
        BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
except Exception:
    BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

if BACKEND == "ollama":
    from llm_ollama_adapter import OllamaLLM as LLMBackend
else:
    from llm_openai_adapter import OpenAILLM as LLMBackend

def _L(event, detail=""):
    try: logger.info(f"{event} {detail}", module="tts")
    except Exception: print(event, detail, flush=True)


class OptimizedKokoroTTS:
    def __init__(self):
        self.word_buffer = []
        self.synthesis_active = False
        self.text_queue, self.audio_queue = Queue(), Queue()
        self.first_word_time = None
        self.cancel_event = threading.Event()  # <- used by Synthesizer to abort work

        from RealtimeTTS import TextToAudioStream, KokoroEngine
        self.engine = KokoroEngine(
            voice="af_bella",
            default_speed=1.0,
            trim_silence=True,
            silence_threshold=0.005,
            extra_start_ms=5, extra_end_ms=5,
            fade_in_ms=2,  fade_out_ms=2,
        )
        self.stream = TextToAudioStream(self.engine, muted=True, playout_chunk_size=256, output_device_index=None)
        # metrics
        self.question_end_time = None
        self.last_q_to_response_ms = None

    def start(self):
        from audio_player import AudioPlayer
        from tts_synthesis import Synthesizer

        def _on_first_play(text):
            if self.first_word_time:
                now = time.time()
                _L("🔊 FIRST SPEECH", f"delay={now-self.first_word_time:.3f}s text='{(text[:50]+'...') if len(text)>50 else text}'")
                # compute QUESTION-TO-RESPONSE_MS if question_end_time was set
                qend = getattr(self, 'question_end_time', None)
                if qend:
                    try:
                        ms = (now - qend) * 1000.0
                        self.last_q_to_response_ms = ms
                    except Exception:
                        self.last_q_to_response_ms = None

        self.audio_player = AudioPlayer(self.audio_queue, logger=_L, on_first_play=_on_first_play)
        self.synth  = Synthesizer(self.stream, self.text_queue, self.audio_queue, state=self, logger=_L)
        self.audio_player.start()
        threading.Thread(target=self.synth.run_worker, daemon=True, name="synth-worker").start()
        _L("🎛️ Pipeline", "started")

    def stop(self):
        try: self.text_queue.join()
        except: pass
        try: self.audio_player.stop(timeout=2.0)
        except: pass
        _L("🛑 Pipeline", "stopped")

    # --------- hard, immediate abort path ---------
    def interrupt_now(self):
        _L("🛑 TTS INTERRUPT", "CANCEL synth, stop audio_player, purge queues")
        try:
            # 1) tell synthesizer to abort ASAP
            self.cancel_event.set()

            # 2) stop audio_player thread and kill ffplay immediately
            try:
                self.audio_player.stop(timeout=0.3)
            except Exception:
                pass

            # 3) purge queues entirely (text + audio)
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait(); self.text_queue.task_done()
                except:
                    break
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break

            self.synthesis_active = False
            self.first_word_time = None

            # 4) start a FRESH audio queue + audio_player
            #    *** CRITICAL: also rebind the synthesizer to this new queue ***
            from audio_player import AudioPlayer
            new_audio_q = Queue()
            self.audio_queue = new_audio_q
            if hasattr(self, "synth") and self.synth is not None:
                self.synth.audio_queue = new_audio_q  # <- the missing link

            self.audio_player = AudioPlayer(self.audio_queue, logger=_L, on_first_play=None)
            self.audio_player.start()
            _L("🔁 Audio route rebound", f"audio_q_id={id(self.audio_queue)} synth_q_id={id(self.synth.audio_queue)}")

        except Exception as e:
            _L("⚠️ Interrupt error", str(e))


    def reset(self):
        while not self.text_queue.empty():
            try: self.text_queue.get_nowait(); self.text_queue.task_done()
            except: break
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except: break
        self.word_buffer.clear()
        self.first_word_time = None
        # allow next utterance to proceed
        self.cancel_event.clear()
        _L("🔄 Pipeline", "reset")

    # --------- LLM streaming input API ---------
    def add_text(self, text: str):
        if not text or not text.strip(): return
        if self.first_word_time is None:
            self.first_word_time = time.time()
            _L("📝 First word", f"'{text.strip()}'")
        self.word_buffer.extend(text.strip().split())
        self._emit_full_sentences_if_any(True)

    def finish_stream(self):
        self._emit_full_sentences_if_any(True)
        if self.word_buffer:
            s = self._join(self.word_buffer)
            if s:
                self.text_queue.put(s); _L("📝 Final", f"'{(s[:60]+'...') if len(s)>60 else s}'")
            self.word_buffer.clear()
        try: self.text_queue.join()
        except: pass
        time.sleep(0.2)

    def _emit_full_sentences_if_any(self, emit_all=False):
        i = 0
        while i < len(self.word_buffer):
            tok = self.word_buffer[i]; term = None
            if tok in ('.','!','?'):
                sent_tokens = self.word_buffer[:i]; sentence = self._join(sent_tokens) + tok; rest = self.word_buffer[i+1:]; term = True
            elif tok.endswith(('.','!','?')):
                sent_tokens = self.word_buffer[:i+1]; sentence = self._join(sent_tokens); rest = self.word_buffer[i+1:]; term = True
            if term:
                s = sentence.strip()
                if s: self.text_queue.put(s); _L("🎭 Sentence", f"'{(s[:60]+'...') if len(s)>60 else s}'")
                self.word_buffer = rest
                if not emit_all: return True
                i = 0; continue
            i += 1
        return False

    @staticmethod
    def _join(tokens):
        if not tokens: return ""
        s = " ".join(tokens); s = " ".join(s.split())
        for p in (".","!","?",",",":",";"): s = s.replace(" " + p, p)
        return s if s.strip().strip(".,!?:;\"'()[]-–—") else ""

    def wait_until_idle(self, timeout=10.0, poll=0.05, grace=0.15):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.text_queue.empty() and self.audio_queue.empty() and not self.synthesis_active:
                time.sleep(grace); return True
            time.sleep(poll)
        return False


def main():
    logger.info("🧪 KOKORO VOICE CHAT SYSTEM", module="main"); logger.info("=" * 50, module="main")
    try:
        # load llm-specific opts and global system prompt (if present)
        # cfg is loaded at module scope above; use globals() to detect it here.
        llm_cfg = cfg.get('llm', {}) if 'cfg' in globals() else {}
        system_prompt = llm_cfg.get('system_prompt') if isinstance(llm_cfg, dict) else None
        # allow system_prompt as either a string or an array of lines; normalize to string
        if isinstance(system_prompt, list):
            try:
                system_prompt = "\n".join(line for line in system_prompt if line is not None)
            except Exception:
                system_prompt = None
        # Prefer structured backends map; fall back to flat opts for compatibility
        backend_opts = {}
        if isinstance(llm_cfg, dict):
            backends_map = llm_cfg.get('backends')
            if isinstance(backends_map, dict) and backends_map.get(BACKEND):
                backend_opts = backends_map.get(BACKEND).get('opts', {}) or {}
            else:
                backend_opts = llm_cfg.get('opts', {}) or {}

        # resolve api_key from api_key_env if provided
        api_key_env = backend_opts.get('api_key_env')
        if api_key_env:
            backend_opts['api_key'] = os.getenv(api_key_env)

        # instantiate adapter with opts and explicit system_prompt (adapters accept and prefer it)
        llm = LLMBackend(system_prompt=system_prompt, **backend_opts)
        tts = OptimizedKokoroTTS(); tts.start()

        from stt_process import KokoroSTT
        realtime_stt = KokoroSTT()

        def _ready(stt_obj, timeout=12.0):
            try:
                temp = getattr(stt_obj, 'temp_file', None); rf = temp + '.ready' if temp else None
                if not rf: return
                t0 = time.time()
                while time.time() - t0 < timeout:
                    if os.path.exists(rf):
                        try: os.unlink(rf)
                        except: pass
                        return
                    time.sleep(0.1)
            except: pass

        _ready(realtime_stt); _L("🎛️ Pipeline", "started — waiting for STT to announce listening readiness")

        banner = False; convo = 0
        while True:
            convo += 1
            logger.info(f"🗣️ === Conversation {convo} ===", module="main")
            received = realtime_stt.get_question()

            if not banner:
                banner = True
                logger.info("🎯🎯🎯 SYSTEM READY - START SPEAKING NOW! 🎯🎯🎯", module="READY")
                logger.info("Press Ctrl+C to exit", module="READY")

            if not received:
                logger.warn("❌ Failed to get question, retrying...", module="main"); continue

            logger.info(f"🤖 Question: '{received}'", module="main")

            # reset per-turn metrics to avoid stale values
            try:
                tts.last_q_to_response_ms = None
            except Exception:
                pass
            try:
                llm.last_llm_ttf_ms = None
                llm.last_llm_total_ms = None
            except Exception:
                pass

            # mark end of user speech (question end) for timing metrics
            try:
                tts.question_end_time = time.time()
            except Exception:
                pass

            interrupted = stream_with_barge_in(llm, tts, received, logger=_L)

            # notify STT we’re done (or interrupted)
            try:
                with open(realtime_stt.temp_file + ".tts_done", "w") as f: f.write("done")
                _L("📩 tts_done signal", realtime_stt.temp_file + ".tts_done")
            except Exception as e:
                logger.warn(f"⚠️ Failed to write tts_done signal: {e}", module="main")

            logger.info("✅ Response complete" + (" (interrupted)" if interrupted else ""), module="main")
            # report metrics: TTS end-to-end and LLM timings (ms)
            try:
                qms = getattr(tts, 'last_q_to_response_ms', None)
                lttf = getattr(llm, 'last_llm_ttf_ms', None)
                ltot = getattr(llm, 'last_llm_total_ms', None)
                if qms is not None:
                    logger.info(f"QUESTION-TO-RESPONSE_MS {qms:.1f}", module="main")
                else:
                    logger.info("QUESTION-TO-RESPONSE_MS unknown", module="main")
                if lttf is not None:
                    logger.info(f"LLM-TTF_MS {lttf:.1f}", module="main")
                else:
                    logger.info("LLM-TTF_MS unknown", module="main")
                if ltot is not None:
                    logger.info(f"LLM-TOTAL_MS {ltot:.1f}", module="main")
                else:
                    logger.info("LLM-TOTAL_MS unknown", module="main")
            except Exception:
                pass
            logger.info("🎙️ Ready for next question...", module="main")
            tts.reset(); time.sleep(0.15)

    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...", module="main")
    finally:
        try: realtime_stt.cleanup()
        except: pass
        try: tts.stop()
        except: pass
        logger.info("✅ Cleanup complete", module="main")


if __name__ == "__main__":
    main()
