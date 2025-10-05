# mic_in_interrupt.py — mic barge-in helper used by pipeline.py
import os, time

try:
    import sounddevice as _sd
except Exception:
    _sd = None


class MicWatcher:
    """
    Minimal RMS gate over microphone. Sets .was_triggered once sustained voice/noise is detected.
    Also supports waiting for QUIET (RMS below threshold) for a sustained window.
    Logs RMS at most every `log_every_s` seconds via the provided logger.
    """
    def __init__(self, logger, thresh=0.018, hold_ms=240, log_every_s=0.2, device=None):
        self.L = logger or (lambda *a, **k: None)
        self.thresh = float(os.getenv("BARGE_IN_THRESH", thresh))
        self.hold_ms = int(os.getenv("BARGE_IN_HOLD_MS", hold_ms))
        self.log_every_s = float(os.getenv("BARGE_IN_LOG_EVERY_S", log_every_s))
        self.device = os.getenv("BARGE_IN_DEVICE", device)

        self.was_triggered = False
        self._started_at = None
        self._stream = None
        self._last_log = 0.0
        self._first_above_logged = False
        self._current_rms = 0.0
        self._now = time.time()

    def _log_rms_throttled(self, rms, now):
        if now - self._last_log >= self.log_every_s:
            meter = "#" * min(20, int(rms * 40))  # crude 0..0.5 scale
            self.L("🎙️ Mic RMS", f"{rms:.3f} (thresh={self.thresh:.3f}) [{meter:20s}]")
            self._last_log = now

    def _cb(self, indata, frames, time_info, status):
        if indata is None:
            return
        try:
            # indata is a NumPy int16 array; avoid importing numpy explicitly
            mv = memoryview(indata.tobytes()).cast('h')
            if not mv:
                return
            s = 0
            n = len(mv)
            for i in range(n):
                s += abs(mv[i])
            rms = (s / max(1, n)) / 32768.0
            now = time.time()

            self._current_rms = rms
            self._now = now

            # periodic "measurement point" logger (<= 5 Hz)
            self._log_rms_throttled(rms, now)

            # voice/noise detection (for barge-in)
            if not self.was_triggered:
                if rms >= self.thresh:
                    if self._started_at is None:
                        self._started_at = now
                        if not self._first_above_logged:
                            self.L("🎚️ Mic above threshold", f"rms={rms:.3f} — arming hold {self.hold_ms}ms")
                            self._first_above_logged = True
                    elif (now - self._started_at) * 1000.0 >= self.hold_ms:
                        self.was_triggered = True
                        self.L("🛎️ BARGE-IN TRIGGERED", f"rms={rms:.3f} held={self.hold_ms}ms")
                else:
                    self._started_at = None
                    self._first_above_logged = False

        except Exception as e:
            self.L("⚠️ MicWatcher cb error", str(e))

    def start(self, samplerate=16000, channels=1):
        if _sd is None:
            self.L("⚠️ MicWatcher disabled", "sounddevice not available")
            return self
        try:
            self.L("🎛️ MicWatcher start",
                   f"thresh={self.thresh:.3f} hold_ms={self.hold_ms} device={self.device}")
            self._stream = _sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype='int16',
                callback=self._cb,
                device=self.device,
                blocksize=0,  # backend decides; our logger limiter enforces <= 5 Hz
            )
            self._stream.start()
        except Exception as e:
            self.L("⚠️ MicWatcher disabled", str(e))
            self._stream = None
        return self

    def stop(self):
        try:
            if self._stream:
                self._stream.stop(); self._stream.close()
                self.L("⏹️ MicWatcher stop", "")
        except Exception as e:
            self.L("⚠️ MicWatcher stop error", str(e))
        self._stream = None

    # -------- NEW: wait for quiet --------
    def wait_for_quiet(self, quiet_ms=600, timeout_s=6.0, below_factor=0.9):
        """
        Block until the mic has been below (thresh * below_factor) for `quiet_ms` consecutively,
        or until `timeout_s` is reached. Returns True if quiet was observed, else False.
        """
        if _sd is None:
            # best effort: just sleep a short while
            time.sleep(min(timeout_s, quiet_ms / 1000.0))
            return True

        target = self.thresh * float(below_factor)
        start_below = None
        t0 = time.time()
        self.L("🤫 Waiting for quiet", f"target<{target:.3f} for {quiet_ms}ms (timeout {timeout_s:.1f}s)")

        while True:
            now = time.time()
            rms = float(self._current_rms)

            if rms < target:
                if start_below is None:
                    start_below = now
                if (now - start_below) * 1000.0 >= quiet_ms:
                    self.L("✅ Quiet window", f"{quiet_ms}ms below {target:.3f}")
                    return True
            else:
                start_below = None

            if now - t0 >= timeout_s:
                self.L("⏱️ Quiet wait timeout", f"last_rms={rms:.3f}")
                return False

            time.sleep(0.02)  # 50 Hz polling


def stream_with_barge_in(llm, tts, question, logger, add_word_delay=0.01,
                         wait_quiet_ms=600, quiet_timeout_s=6.0) -> bool:
    """
    Drive LLM -> TTS with mic barge-in monitoring, including playback phase.
    On interruption, DO NOT generate an answer to the interruption:
      - cut TTS,
      - wait for quiet,
      - return (pipeline will re-arm STT after quiet).
    Returns True if interrupted, False otherwise.
    """
    L = logger or (lambda *a, **k: None)
    interrupted = False
    words_streamed = 0
    watcher = MicWatcher(logger=L).start()

    # --- streaming phase ---
    for w in llm.ask_question(question):
        if watcher.was_triggered:
            interrupted = True
            L("🛑 BARGE-IN (stream)", f"after {words_streamed} words — cutting TTS immediately")
            tts.interrupt_now()
            break
        tts.add_text(w + ' ')
        words_streamed += 1
        time.sleep(add_word_delay)

    # --- playback phase (keep watcher alive) ---
    if not interrupted:
        tts.finish_stream()
        L("⏳ Post-stream playback", "watching mic for barge-in during audio output")
        start_wait = time.time()
        while True:
            if watcher.was_triggered:
                interrupted = True
                elapsed = time.time() - start_wait
                L("🛑 BARGE-IN (playback)", f"after {elapsed:.2f}s of playback — cutting TTS now")
                tts.interrupt_now()
                break
            if tts.text_queue.empty() and tts.audio_queue.empty() and not tts.synthesis_active:
                break
            time.sleep(0.05)
        if not interrupted:
            tts.wait_until_idle(timeout=5.0, poll=0.05, grace=0.20)

    # --- after interruption: WAIT FOR QUIET BEFORE returning ---
    if interrupted:
        # Use the same live mic watcher to wait for quiet
        watcher.wait_for_quiet(quiet_ms=wait_quiet_ms, timeout_s=quiet_timeout_s, below_factor=0.9)
        L("🔁 Post-interrupt", "quiet detected (or timeout) — returning to pipeline WITHOUT answering")

    # cleanup watcher
    try:
        watcher.stop()
    except Exception:
        pass

    return interrupted
