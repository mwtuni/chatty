# kokoro_stt.py - Minimal persistent STT in separate process
"""🎙️ Minimal STT that runs in separate process and stays alive"""
import subprocess
import tempfile
import json
import os
import sys
import time
import atexit
import logger

class KokoroSTT:
    def __init__(self):
        self.process = None
        self.temp_file = None
        self._start_process()
        atexit.register(self.cleanup)
    
    def _start_process(self):
        """Start the persistent STT process"""
        # Create communication file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.temp_file = f.name
        
        # Start STT worker
        self.process = subprocess.Popen([
            sys.executable, '-c', f'''
import sys
sys.path.append(r"{os.path.dirname(__file__)}")
from realtime_stt import MinimalSTT
import json
import time
import os
import datetime
import threading

def log_info(msg):
    """Subprocess logging with thread info"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = threading.get_ident()
    print(f"[{{timestamp}}] [stt_process] [thread {{thread_id}}] [info] {{msg}}")

def main():
    log_info("🎙️ STT Process: Starting...")
    realtime_stt = None  # Don't create STT yet - lazy initialization
    question_count = 0
    current_question = None
    shutdown_file = r"{self.temp_file}.shutdown"

    # Callback for when STT captures text
    def on_final(text):
        nonlocal current_question
        if current_question is None:  # Only capture if we're listening for a question
            current_question = text.strip()

    try:
        # Signal that subprocess is ready (but STT not started yet)
        with open(r"{self.temp_file}.ready", "w") as f:
            f.write("ready")
        log_info("🎙️ STT Process: Ready (STT not started yet - waiting for first listen command)")

        while True:
            # If main requested shutdown, exit cleanly
            if os.path.exists(shutdown_file):
                log_info("🎙️ STT Process: Shutdown requested, exiting loop...")
                break

            # Wait for listen command before capturing next question
            listen_file = r"{self.temp_file}.listen"
            log_info(f"🎙️ STT Process: Waiting for listen command...")
            while not os.path.exists(listen_file):
                # check for shutdown while waiting
                if os.path.exists(shutdown_file):
                    log_info("🎙️ STT Process: Shutdown requested while waiting, exiting...")
                    break
                time.sleep(0.1)

            if os.path.exists(shutdown_file):
                break

            # Initialize STT on first use (lazy initialization after TTS is ready)
            if realtime_stt is None:
                log_info("🎙️ STT Process: First listen command - initializing STT now...")
                realtime_stt = MinimalSTT(language="en")
                realtime_stt.on_final_text = on_final
                realtime_stt.start()
                log_info("🎙️ STT Process: STT initialized and listening started")

            # Remove listen command file
            try:
                os.unlink(listen_file)
            except:
                pass

            question_count += 1
            log_info(f"🎙️ STT Process: Ready for question {{question_count}} (listening...)")
            # Print a clear banner so the parent/console shows STT is actively listening
            print("\033[1;32m=== STT READY: Listening for questions ===\033[0m", flush=True)

            try:
                # Reset and wait for next question
                current_question = None

                # Wait for question to be captured
                while not current_question:
                    if os.path.exists(shutdown_file):
                        log_info("🎙️ STT Process: Shutdown requested while capturing, exiting...")
                        break
                    time.sleep(0.1)

                if os.path.exists(shutdown_file):
                    break

                log_info(f"🎙️ STT Process: Got question {{question_count}}: '{{current_question}}'")

                # Temporarily pause STT processing to free GPU resources for TTS
                # Use abort_generation() to stop current processing without tearing down pipes
                log_info("🎙️ STT Process: Pausing (abort current processing) for TTS...")
                try:
                    realtime_stt.abort_generation()
                except Exception:
                    # abort_generation might not exist, that's ok
                    pass

                # Write to file
                with open(r"{self.temp_file}", "w") as f:
                    json.dump({{"question": current_question, "count": question_count}}, f)

                # Wait for TTS completion signal before resuming
                tts_done_file = r"{self.temp_file}.tts_done"
                while not os.path.exists(tts_done_file):
                    if os.path.exists(shutdown_file):
                        break
                    time.sleep(0.2)

                # Clean up signal and resume STT listening if not shutting down
                try:
                    if os.path.exists(tts_done_file):
                        os.unlink(tts_done_file)
                except:
                    pass

                log_info("🎙️ STT Process: Resuming listening (already running)...")

            except Exception as e:
                log_info(f"🎙️ STT Process: Error: {{e}}")
                time.sleep(2)

    finally:
        # Ensure recorder is stopped cleanly on exit
        try:
            if realtime_stt is not None:
                log_info("🎙️ STT Process: Cleaning up STT and exiting...")
                realtime_stt.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
'''
        ])
        
        # Wait for startup and ready signal
        time.sleep(3)
        
        # Wait for STT to signal it's ready
        ready_file = self.temp_file + ".ready"
        timeout = 0
        while not os.path.exists(ready_file) and timeout < 30:  # 30 second timeout
            time.sleep(0.5)
            timeout += 0.5
            
        if os.path.exists(ready_file):
            try:
                os.unlink(ready_file)  # Clean up ready signal
            except:
                pass
            logger.info("🎙️ Persistent STT started", module="realtime_stt")
        else:
            logger.warn("⚠️ STT startup timeout, continuing anyway...", module="realtime_stt")
    
    def get_question(self):
        """Get next question from persistent process"""
        if not self.process or self.process.poll() is not None:
            logger.info("🎙️ STT process died, restarting...", module="realtime_stt")
            self._start_process()
        
        # Signal STT to listen for next question
        listen_file = self.temp_file + ".listen"
        with open(listen_file, 'w') as f:
            f.write("listen")
        
        logger.info("🎙️ Waiting for voice input...", module="realtime_stt")
        
        # Read current state if any
        last_count = 0
        try:
            if os.path.exists(self.temp_file):
                with open(self.temp_file, 'r') as f:
                    data = json.load(f)
                    last_count = data.get('count', 0)
        except:
            pass
        
        # Wait for new question
        max_wait = 120
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                if os.path.exists(self.temp_file):
                    with open(self.temp_file, 'r') as f:
                        data = json.load(f)
                    
                    current_count = data.get('count', 0)
                    if current_count > last_count:
                        return data['question']
                        
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            
            time.sleep(0.5)
            wait_time += 0.5
        
        return None
    
    def pause_for_tts(self):
        """Signal STT to pause (abort generation) to free GPU for TTS"""
        if not self.temp_file:
            return
        pause_file = self.temp_file + ".pause"
        try:
            with open(pause_file, 'w') as f:
                f.write("pause")
            logger.debug("Sent pause signal to STT", module="realtime_stt")
        except Exception as e:
            logger.warn(f"Failed to send pause signal: {e}", module="realtime_stt")
    
    def resume_from_tts(self):
        """Signal STT to resume after TTS is done"""
        if not self.temp_file:
            return
        pause_file = self.temp_file + ".pause"
        try:
            if os.path.exists(pause_file):
                os.unlink(pause_file)
            logger.debug("Removed pause signal from STT", module="realtime_stt")
        except Exception as e:
            logger.warn(f"Failed to remove pause signal: {e}", module="realtime_stt")
    
    def cleanup(self):
        """Clean up process"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                try:
                    self.process.kill()
                except:
                    pass
        
        # Remove temp files and signals
        if self.temp_file:
            try:
                # Remove the main json file
                os.unlink(self.temp_file)
            except:
                pass
            try:
                # Remove ready and tts_done signals if exist
                ready = self.temp_file + ".ready"
                if os.path.exists(ready):
                    os.unlink(ready)
            except:
                pass
            try:
                tts_done = self.temp_file + ".tts_done"
                if os.path.exists(tts_done):
                    os.unlink(tts_done)
            except:
                pass
            try:
                pause = self.temp_file + ".pause"
                if os.path.exists(pause):
                    os.unlink(pause)
            except:
                pass

if __name__ == "__main__":
    realtime_stt = KokoroSTT()
    for i in range(3):
        question = realtime_stt.get_question()
        logger.info(f"Main: Got '{question}'", module="realtime_stt")