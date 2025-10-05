# audio_player.py — ffplay sink for raw S16LE; Windows-safe, low-latency (hard-stop safe)
import sys, os, time, threading, subprocess
from queue import Empty

RATE, CH, SW = 44100, 2, 2  # 44.1k, stereo, s16le


class AudioPlayer:
    def __init__(self, audio_queue, logger=None, on_first_play=None,
                 ffplay_path="ffplay", fade_ms=6, sdl_driver=None, prime_ms=25, head_pad_ms=None):
        self.q = audio_queue
        self.L = logger or (lambda *a, **k: None)
        self.cb = on_first_play
        self.ffplay = ffplay_path
        self.fade_ms = int(fade_ms) if fade_ms else 0
        self.sdl_driver = sdl_driver
        self.prime_ms = max(0, int(prime_ms))
        self.head_pad_ms = 160 if head_pad_ms is None and sys.platform.startswith("win") else (head_pad_ms or 0)

        self._run = False
        self._stopping = False
        self._t = None
        self._proc = None
        self._stdin = None
        self._first_cb_fired = False
        self._last_nonempty_label = ""
        self._supports_ac = True
        self._out_channels = CH
        self._downmix_to_mono = False
        self._head_pad_done = False

    def _log(self, e, d=""):
        try: self.L(e, d)
        except Exception: pass

    # ---------- lifecycle ----------
    def start(self):
        if self._run: return
        self._stopping = False
        self._spawn_ffplay(capture_stderr=True)
        self._run = True
        self._t = threading.Thread(target=self._worker, daemon=True, name="ffplay-writer")
        self._t.start()
        self._log("▶️ AudioPlayer.start", "ffplay worker started")

    def stop(self, timeout=0.3):
        # mark stopping first so the worker knows NOT to respawn
        self._stopping = True
        self._run = False
        try:
            if self._stdin and not self._stdin.closed:
                self._stdin.flush(); self._stdin.close()
        except Exception: pass
        self._kill_ffplay()  # kill quickly to avoid trailing audio
        if self._t:
            try: self._t.join(timeout=timeout)
            except Exception: pass
        self._head_pad_done = False
        self._first_cb_fired = False
        self._log("⏹️ AudioPlayer.stop", f"joined thread timeout={timeout}")

    # ---------- ffplay ----------
    def _build_cmd(self, with_ac: bool):
        cmd = [
            self.ffplay,
            "-hide_banner", "-loglevel", "warning", "-nostats",
            "-nodisp", "-autoexit",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-blocksize", "4096",
            "-max_delay", "0",
            "-sync", "audio",
            "-f", "s16le",
            "-ar", str(RATE),
        ]
        af = ["aresample=async=1:min_hard_comp=0.100:first_pts=0"]
        if with_ac:
            cmd += ["-ac", "2"]
        else:
            af.append("pan=stereo|c0=c0|c1=c0")
        cmd += ["-af", ",".join(af), "-i", "-"]
        return cmd

    def _spawn_ffplay(self, capture_stderr=False):
        if self._stopping: return  # never spawn while stopping
        try_with_ac = self._supports_ac
        cmd = self._build_cmd(try_with_ac)
        self._out_channels = 2 if try_with_ac else 1
        self._downmix_to_mono = self._out_channels == 1

        self._log("ℹ️ starting ffplay", " ".join(cmd))
        stderr = subprocess.PIPE if capture_stderr else subprocess.DEVNULL
        env = os.environ.copy()
        if self.sdl_driver: env["SDL_AUDIODRIVER"] = self.sdl_driver
        else: env.setdefault("SDL_AUDIODRIVER", "wasapi")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr,
            bufsize=0, creationflags=creationflags, env=env,
        )
        self._stdin = self._proc.stdin

        # Probe: if -ac unsupported, respawn without it
        if capture_stderr:
            time.sleep(0.12)
            if self._proc.poll() is not None:
                try: err = (self._proc.stderr.read() or b"").decode("utf-8", errors="replace").strip()
                except Exception: err = ""
                self._log("⚠️ ffplay exited early", f"rc={self._proc.returncode} stderr={err}")
                if "option 'ac'" in err.lower() or "option not found" in err.lower() or "failed to set value" in err.lower():
                    self._supports_ac = False
                    self._log("⚙️ fallback", "ffplay lacks -ac; switching to mono input + pan upmix")
                    if self._stopping: return
                    cmd2 = self._build_cmd(with_ac=False)
                    self._out_channels = 1; self._downmix_to_mono = True
                    self._log("ℹ️ retry ffplay", " ".join(cmd2))
                    self._proc = subprocess.Popen(
                        cmd2, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        bufsize=0, creationflags=creationflags, env=env,
                    )
                    self._stdin = self._proc.stdin
                    time.sleep(0.05)

        # Prime a touch of silence so playback starts instantly
        if (not self._stopping) and self._stdin and not self._stdin.closed and self.prime_ms > 0:
            try:
                frames = int(RATE * self.prime_ms / 1000.0)
                self._stdin.write(b"\x00" * (frames * self._out_channels * SW)); self._stdin.flush()
            except Exception: pass

    def _kill_ffplay(self):
        # hard stop without respawn
        try:
            if self._stdin and not self._stdin.closed:
                self._stdin.flush(); self._stdin.close()
        except Exception: pass
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.25)
            except Exception:
                try: self._proc.kill()
                except Exception: pass
        self._proc = None; self._stdin = None

    # ---------- helpers ----------
    @staticmethod
    def _stereo_to_mono(pcm: bytes) -> bytes:
        if not pcm: return pcm
        if len(pcm) % 4 != 0: pcm = pcm[: len(pcm) // 4 * 4]
        out = bytearray(len(pcm) // 2); mv = memoryview(pcm); o = 0
        for i in range(0, len(mv), 4):
            l = int.from_bytes(mv[i:i+2], "little", signed=True)
            r = int.from_bytes(mv[i+2:i+4], "little", signed=True)
            m = (l + r) // 2
            out[o:o+2] = int(m).to_bytes(2, "little", signed=True); o += 2
        return bytes(out)

    def _apply_fades(self, pcm: bytes, ms: int, ch: int) -> bytes:
        if ms <= 0 or not pcm: return pcm
        frames = len(pcm) // (ch * SW)
        if frames <= 0: return pcm
        fade = int(RATE * ms / 1000.0)
        if fade <= 0 or frames <= fade * 2: return pcm
        ba = bytearray(pcm)
        def scale(i_bytes, f):
            s = int.from_bytes(ba[i_bytes:i_bytes+2], "little", signed=True)
            s = max(-32768, min(32767, int(s * f)))
            ba[i_bytes:i_bytes+2] = int(s).to_bytes(2, "little", signed=True)
        for i in range(fade):
            f = i / float(fade)
            off = i * ch * SW
            for c in range(ch): scale(off + c * SW, f)
        start = frames - fade
        for i in range(fade):
            f = (fade - 1 - i) / float(fade)
            off = (start + i) * ch * SW
            for c in range(ch): scale(off + c * SW, f)
        return bytes(ba)

    # ---------- worker ----------
    def _worker(self):
        sess = 0
        while True:
            # exit quickly if stopping; also DRAIN queue to avoid ghost audio
            if not self._run or self._stopping:
                try:
                    while True:
                        self.q.get_nowait()
                        self.q.task_done()
                except Exception:
                    pass
                break

            try:
                pcm, label = self.q.get(timeout=0.2)
            except Empty:
                continue

            if self._stopping:
                self.q.task_done()
                break

            lab = (label or "").strip()
            if lab and not self._first_cb_fired and self.cb:
                self._first_cb_fired = True
                try: self.cb(lab)
                except Exception: pass
            if lab: self._last_nonempty_label = lab

            if (self._proc is None) or (self._proc.poll() is not None):
                if self._stopping:
                    self.q.task_done(); break
                self._spawn_ffplay()

            if self._downmix_to_mono:
                pcm_to_write = self._stereo_to_mono(pcm); ch_out = 1
            else:
                pcm_to_write = pcm; ch_out = 2

            if not self._head_pad_done and self.head_pad_ms > 0 and not self._stopping:
                try:
                    frames = int(RATE * self.head_pad_ms / 1000.0)
                    self._stdin.write(b"\x00" * (frames * ch_out * SW)); self._stdin.flush()
                except Exception as e:
                    self._log("⚠️ head_pad write failed", str(e))
                self._head_pad_done = True

            if self.fade_ms: pcm_to_write = self._apply_fades(pcm_to_write, self.fade_ms, ch_out)

            sess += 1
            try:
                if self._stopping:
                    self.q.task_done(); break
                self._stdin.write(pcm_to_write); self._stdin.flush()
            except Exception as e:
                if self._stopping or not self._run:
                    self._log("⚠️ write failed", f"sess={sess} {e}")
                    self.q.task_done()
                    break
                self._log("⚠️ write failed", f"sess={sess} {e}")
                self._log("❌ ffplay pipe broken", "respawning")
                self._kill_ffplay()
                if self._stopping:
                    self.q.task_done(); break
                self._spawn_ffplay()
                try:
                    if self._stopping:
                        self.q.task_done(); break
                    self._stdin.write(pcm_to_write); self._stdin.flush()
                except Exception:
                    self._log("❌ drop chunk", "gave up after retry")
                    self.q.task_done()
                    if self._stopping: break
                    continue

            dur = (len(pcm_to_write) // (ch_out * SW)) / float(RATE)
            preview = (self._last_nonempty_label[:40] + "...") if (self._last_nonempty_label and len(self._last_nonempty_label) > 40) else (self._last_nonempty_label or "")
            self._log("▶️ play_session", f"sess={sess} lenB={len(pcm_to_write)} dur_s={dur:.3f} out_ch={ch_out} text='{preview}'")
            self.q.task_done()
