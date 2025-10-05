#!/usr/bin/env python3
# tts_synthesis.py — stream TTS audio OR coalesce full sentences before enqueue
# - Windows-safe chunking (bigger coalesce & timeout, flush floor)
# - Optional full-sentence rendering (default on Windows)
# - Inter-sentence silence for natural rhythm
# - **Immediate cancel support** via state.cancel_event (drop & abort mid-tts_synthesis)

import time, sys, audioop, os
from queue import Empty

# --- Engine native format ---
SRC_RATE = 24000
SRC_CH = 1
BYTES_PER_SAMPLE = 2
SRC_FRAME_BYTES = BYTES_PER_SAMPLE * SRC_CH

# --- Canonical playback format ---
RATE = 44100
CH = 2
CHANNELS = CH
FRAME_BYTES = BYTES_PER_SAMPLE * CH

# --- Platform tuning ---
WIN = sys.platform.startswith("win")

# Coalescing target and flush timeout
MIN_MS = 300 if WIN else 220
FLUSH_TIMEOUT = 0.12 if WIN else 0.08

# Never flush below this floor even on timeout (prevents 5ms sneeze packets)
MIN_FLOOR_MS = 140 if WIN else 80

MIN_BYTES = int(RATE * FRAME_BYTES * (MIN_MS / 1000.0))
MIN_FLOOR_BYTES = int(RATE * FRAME_BYTES * (MIN_FLOOR_MS / 1000.0))

# Inter-sentence pause (ms) to avoid butt-joined sentences
INTER_SENTENCE_MS = int(os.environ.get("TTS_INTER_SENTENCE_MS", "160"))  # try 140–220


# ---------------- Conversion helper ----------------
class Upconverter:
    """Convert 24 kHz mono -> 44.1 kHz stereo, stateful across chunks."""
    def __init__(self):
        self._state = None

    def reset(self):
        self._state = None

    def convert(self, pcm_mono_24k: bytes) -> bytes:
        if not pcm_mono_24k:
            return b""
        # Resample 24k → 44.1k with continuity across calls
        out, self._state = audioop.ratecv(
            pcm_mono_24k, BYTES_PER_SAMPLE, SRC_CH, SRC_RATE, RATE, self._state
        )
        # Duplicate mono → stereo (L=R)
        return audioop.tostereo(out, BYTES_PER_SAMPLE, 1.0, 1.0)


# ---------------- Delta PCM coalescer ----------------
class _DeltaPCM:
    def __init__(self, min_bytes=MIN_BYTES, flush_timeout=FLUSH_TIMEOUT, logger=lambda *a, **k: None):
        self._sent_total = 0
        self._buf = bytearray()
        self._min_bytes = int(min_bytes)
        self._flush_timeout = float(flush_timeout)
        self._last_flush = time.time()
        self._log = logger

    def reset(self):
        self._sent_total = 0
        self._buf.clear()
        self._last_flush = time.time()

    def offer_full(self, full_pcm: bytes) -> bytes:
        if not full_pcm:
            return b""

        # Only append the new tail of a growing "full" buffer
        if len(full_pcm) > self._sent_total:
            tail = full_pcm[self._sent_total:]
            self._sent_total = len(full_pcm)
        else:
            tail = full_pcm

        if tail:
            self._buf.extend(tail)

        now = time.time()
        # Flush if big enough, or timed out AND buffer reached a safe floor size
        if len(self._buf) >= self._min_bytes or (
            (now - self._last_flush) >= self._flush_timeout and len(self._buf) >= MIN_FLOOR_BYTES
        ):
            out = bytes(self._buf)
            if out:
                dur = len(out) / (RATE * FRAME_BYTES)
                self._log("🔉 Enqueued", f"bytes={len(out)} dur_s={dur:.3f}")
                self._buf.clear()
                self._last_flush = now
                return out

        return b""

    def flush(self) -> bytes:
        if not self._buf:
            return b""
        out = bytes(self._buf)
        self._buf.clear()
        dur = len(out) / (RATE * FRAME_BYTES)
        self._log("🔉 Enqueued", f"bytes={len(out)} dur_s={dur:.3f}")
        self._last_flush = time.time()
        return out


# ---------------- Synthesizer ----------------
class Synthesizer:
    def __init__(self, stream, text_queue, audio_queue, state=None, logger=None,
                 full_sentence_mode=None, sentence_slice_ms=0, inter_sentence_ms=INTER_SENTENCE_MS):
        """
        full_sentence_mode:
            True  -> buffer entire Kokoro sentence, upconvert once, enqueue one big chunk
            False -> stream via _DeltaPCM (~220–300 ms chunks)
            None  -> auto (True on Windows, False elsewhere)
        sentence_slice_ms:
            If >0, split a full sentence into ~N ms slabs when enqueueing (e.g., 400).
        inter_sentence_ms:
            Silence inserted after each sentence for natural rhythm.
        """
        self.stream = stream
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.state = state
        self._log = logger or (lambda *a, **k: None)

        self._delta = _DeltaPCM(logger=self._log)
        self._cur_label = ""
        self._up = Upconverter()

        # Defaults: Windows prefers full sentence for smoothness
        self.full_sentence_mode = (WIN if full_sentence_mode is None else bool(full_sentence_mode))
        self.sentence_slice_ms = int(sentence_slice_ms or 0)
        self._gap_ms = max(0, int(inter_sentence_ms))

    # ---- cancel helpers ----
    def _canceled(self) -> bool:
        ce = getattr(self.state, "cancel_event", None)
        return bool(ce and ce.is_set())

    def _abort_stream_now(self):
        """Best-effort stop of the underlying stream/engine."""
        s = getattr(self, "stream", None)
        for m in ("interrupt", "stop", "flush", "reset", "cancel"):
            try:
                fn = getattr(s, m, None)
                if callable(fn):
                    fn()
                    break
            except Exception:
                pass

    # ---- emit helpers ----
    def _emit_pcm_streaming(self, pcm_like: bytes):
        if not pcm_like or self._canceled():
            return
        out44 = self._up.convert(pcm_like)
        if self._canceled():
            return
        tail = self._delta.offer_full(out44)
        if tail and not self._canceled():
            self.audio_queue.put((tail, self._cur_label))

    def _emit_sentence_full(self, text: str):
        """Render the whole sentence, upconvert once, then enqueue (optionally sliced)."""
        self._up.reset()
        acc = bytearray()
        aborted = False

        def _on_chunk(chunk: bytes):
            nonlocal aborted
            if self._canceled():
                aborted = True
                return  # drop further audio immediately
            if chunk:
                acc.extend(chunk)

        # Synthesize full sentence
        self.stream.feed(text)
        try:
            self.stream.play(
                log_synthesized_text=False,
                on_audio_chunk=_on_chunk,
                muted=True,
                fast_sentence_fragment=False,
                force_first_fragment_after_words=999999,
            )
        except Exception as e:
            self._log("⚠️ Synthesis error", str(e))
            aborted = True

        if self._canceled() or aborted:
            self._abort_stream_now()
            return  # do not enqueue anything

        if not acc:
            return

        # Upconvert the full sentence in one go
        pcm44 = self._up.convert(bytes(acc))
        if self._canceled():
            self._abort_stream_now()
            return

        # Optionally slice into large slabs (e.g., 400–600 ms) to keep pipe activity steady
        if self.sentence_slice_ms and self.sentence_slice_ms > 0:
            stride = int(RATE * FRAME_BYTES * (self.sentence_slice_ms / 1000.0))
            if stride < 1:
                stride = len(pcm44)
            for i in range(0, len(pcm44), stride):
                if self._canceled():
                    self._abort_stream_now()
                    return
                self.audio_queue.put((pcm44[i:i+stride], self._cur_label))
        else:
            if not self._canceled():
                self.audio_queue.put((pcm44, self._cur_label))

        # Inter-sentence gap for natural rhythm
        self._emit_silence(self._gap_ms)

    def _emit_silence(self, ms: int):
        if ms <= 0 or self._canceled():
            return
        frames = int(RATE * ms / 1000.0)
        if frames <= 0:
            return
        self.audio_queue.put((b"\x00" * (frames * FRAME_BYTES), ""))

    # ---- worker ----
    def run_worker(self):
        while True:
            # If interrupted between sentences, flush engine and idle
            if self._canceled():
                self._abort_stream_now()
                time.sleep(0.01)
                continue

            try:
                text = self.text_queue.get(timeout=1.0)
                label = (text or "").strip()
                if not label:
                    self.text_queue.task_done()
                    continue

                self._cur_label = label
                if self.state is not None:
                    setattr(self.state, "synthesis_active", True)

                # Fresh coalescer & resampler state per sentence
                self._delta.reset()
                self._up.reset()

                try:
                    if self.full_sentence_mode:
                        # One big render per complete sentence
                        if not self._canceled():
                            self._emit_sentence_full(label)
                    else:
                        # Streamed render with safe coalescing
                        def _on_chunk(chunk: bytes):
                            # Fast drop if interrupted mid-stream
                            if self._canceled():
                                return
                            self._emit_pcm_streaming(chunk)

                        self.stream.feed(label)
                        if not self._canceled():
                            self.stream.play(
                                log_synthesized_text=False,
                                on_audio_chunk=_on_chunk,
                                muted=True,
                                fast_sentence_fragment=False,
                                force_first_fragment_after_words=999999,
                            )

                        # Final fragment for this sentence
                        if not self._canceled():
                            tail = self._delta.flush()
                            if tail:
                                self.audio_queue.put((tail, self._cur_label))

                        # Inter-sentence gap
                        self._emit_silence(self._gap_ms)

                except Exception as e:
                    self._log("⚠️ Synthesis error", str(e))
                finally:
                    if self.state is not None:
                        setattr(self.state, "synthesis_active", False)
                self.text_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                self._log("❌ Synthesis worker failed", str(e))
                continue

    def synthesis_worker(self):
        return self.run_worker()
