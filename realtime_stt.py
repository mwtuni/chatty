# realtime_stt.py - Minimal Speech-to-Text Implementation
import threading
import time
from typing import Optional, Callable
from RealtimeSTT import AudioToTextRecorder
import logger

class MinimalSTT:
    """
    Minimal Speech-to-Text implementation using RealtimeSTT
    Optimized for low latency with essential features only
    """
    
    def __init__(self, 
                 language: str = "en",
                 on_realtime_text: Optional[Callable[[str], None]] = None,
                 on_final_text: Optional[Callable[[str], None]] = None,
                 on_silence_start: Optional[Callable[[], None]] = None):
        """
        Initialize minimal STT with callbacks
        
        Args:
            language: Language code for transcription
            on_realtime_text: Callback for partial transcription
            on_final_text: Callback for complete transcription  
            on_silence_start: Callback when silence starts
        """
        self.language = language
        self.on_realtime_text = on_realtime_text
        self.on_final_text = on_final_text
        self.on_silence_start = on_silence_start
        
        # Minimal config optimized for speed
        self.config = {
            "model": "base.en",
            "language": language,
            "use_microphone": True,
            "spinner": False,
            "realtime_model_type": "base.en",
            "silero_sensitivity": 0.05,
            "webrtc_sensitivity": 3,
            "post_speech_silence_duration": 0.7,  # Short timeout for responsiveness
            "min_length_of_recording": 0.5,
            "realtime_processing_pause": 0.03,   # Minimal delay
            "enable_realtime_transcription": True,
            "beam_size": 3,
            "beam_size_realtime": 3,
            "no_log_file": True,
        }
        
        self.recorder = None
        self.is_running = False
        self.current_text = ""
        
    def start(self):
        """Start the STT recorder"""
        if self.is_running:
            return
            
        logger.info("Starting speech recognition...")
        
        self.recorder = AudioToTextRecorder(
            model=self.config["model"],
            language=self.config["language"],
            spinner=self.config["spinner"],
            use_microphone=self.config["use_microphone"],
            realtime_model_type=self.config["realtime_model_type"],
            silero_sensitivity=self.config["silero_sensitivity"],
            webrtc_sensitivity=self.config["webrtc_sensitivity"],
            post_speech_silence_duration=self.config["post_speech_silence_duration"],
            min_length_of_recording=self.config["min_length_of_recording"],
            realtime_processing_pause=self.config["realtime_processing_pause"],
            enable_realtime_transcription=self.config["enable_realtime_transcription"],
            beam_size=self.config["beam_size"],
            beam_size_realtime=self.config["beam_size_realtime"],
            no_log_file=self.config["no_log_file"],
        )
        
        # Set up callbacks
        self.recorder.realtime_transcription_callback = self._on_realtime_transcription
        self.recorder.transcription_callback = self._on_final_transcription
        self.recorder.recording_start_callback = self._on_recording_start
        
        self.is_running = True
        
        # Start listening loop in background thread
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
    def stop(self):
        """Stop the STT recorder"""
        if not self.is_running:
            return
            
        logger.info("Stopping speech recognition...")
        self.is_running = False
        # Attempt to shutdown recorder safely
        try:
            if self.recorder:
                self.recorder.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down recorder: {e}")

        # Wait for listen thread to finish
        try:
            if hasattr(self, 'listen_thread') and self.listen_thread.is_alive():
                self.listen_thread.join(timeout=2.0)
        except Exception as e:
            logger.error(f"Error joining listen thread: {e}")
            
    def _listen_loop(self):
        """Main listening loop"""
        while self.is_running and self.recorder:
            try:
                # This call blocks until speech is detected and processed
                text = None
                try:
                    text = self.recorder.text()
                except BrokenPipeError as bp:
                    # Under Windows, pipe can be closed if child process exits.
                    logger.error(f"Broken pipe in recorder.text(): {bp}")
                    # Attempt a graceful shutdown and exit loop
                    self.is_running = False
                    break
                except OSError as oe:
                    # Catch low-level I/O errors from the connection
                    logger.error(f"OSError in recorder.text(): {oe}")
                    self.is_running = False
                    break

                if text and text.strip():
                    logger.debug(f"Final from loop: {text}")
                    if self.on_final_text:
                        self.on_final_text(text.strip())
                        
            except Exception as e:
                # Log unexpected errors but keep loop alive briefly
                logger.error(f"Error in listen loop: {e}")
                if self.is_running:  # Only sleep if still running
                    time.sleep(0.1)
                    
    def _on_realtime_transcription(self, text: str):
        """Handle real-time transcription updates"""
        if text and text != self.current_text:
            self.current_text = text
            logger.debug(f"Partial: {text}")
            if self.on_realtime_text:
                self.on_realtime_text(text)
                
    def _on_final_transcription(self, text: str):
        """Handle final transcription result"""
        if text and text.strip():
            logger.debug(f"Complete: {text}")
            self.current_text = ""
            if self.on_final_text:
                self.on_final_text(text.strip())
                
    def _on_recording_start(self):
        """Handle recording start event"""
        logger.debug("Recording started...")
        if self.on_silence_start:
            self.on_silence_start()
            
    def abort_generation(self):
        """Abort any ongoing processing"""
        if self.recorder:
            # Stop current processing
            self.recorder.abort()
            
    def get_current_text(self) -> str:
        """Get current partial transcription"""
        return self.current_text


if __name__ == "__main__":
    # Test the STT module
    def on_partial(text):
        logger.info(f"Partial: {text}", module="stt_test")
        
    def on_final(text):
        logger.info(f"Final: {text}", module="stt_test")
        
    realtime_stt = MinimalSTT(
        on_realtime_text=on_partial,
        on_final_text=on_final
    )
    
    try:
        realtime_stt.start()
        logger.info("Speak into your microphone. Press Ctrl+C to stop.", module="stt_test")
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\nStopping...", module="stt_test")
        realtime_stt.stop()