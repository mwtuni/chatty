# llm.py - Minimal LLM Implementation with OpenAI API
import os
from typing import Optional, Generator, List, Dict
from openai import OpenAI
import time
import logger

class MinimalLLM:
    """
    Minimal LLM implementation using OpenAI API
    Optimized for streaming responses with low latency
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 system_prompt: str = "You are a helpful assistant. Be concise and conversational."):
        """
        Initialize minimal LLM with OpenAI API
        
        Args:
            api_key: OpenAI API key (uses env var if None)
            model: Model to use for generation
            system_prompt: System prompt for the assistant
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.system_prompt = system_prompt
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Performance tracking
        self.last_request_time = 0
        
        logger.info(f"🧠 Initialized LLM with model: {self.model}", module="llm_base")
        
    def prewarm(self):
        """Prewarm the model to reduce first request latency"""
        logger.info("Prewarming LLM...")
        try:
            # Send a minimal request to initialize connection
            dummy_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=dummy_messages,
                max_tokens=1,
                temperature=0.7
            )
            logger.info("LLM prewarmed successfully")
        except Exception as e:
            logger.error(f"Prewarm failed: {e}")
            
    def stream_response(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate streaming response from user input
        
        Args:
            user_input: User's message
            
        Yields:
            str: Chunks of the response as they're generated
        """
        # Ensure the system message is always the first message (avoid drift)
        try:
            if not self.messages or self.messages[0].get("role") != "system" or self.messages[0].get("content") != self.system_prompt:
                # Replace or insert the canonical system message
                # keep any existing non-system history after it
                non_system = [m for m in (self.messages or []) if m.get("role") != "system"]
                self.messages = [{"role": "system", "content": self.system_prompt}] + non_system
        except Exception:
            # fallback: ensure messages list has at least the system message
            self.messages = [{"role": "system", "content": self.system_prompt}]

        # Add user message to conversation
        self.messages.append({"role": "user", "content": user_input})
        
        start_time = time.time()
        first_token_time = None
        total_tokens = 0
        
        try:
            logger.debug(f"Generating response for: {user_input[:50]}...")
            
            # Create streaming completion
            logger.info(f"🧠 LLM: Creating OpenAI stream for: '{user_input[:30]}...'")
            # Log summary of messages being sent to OpenAI (includes system prompt) at INFO
            try:
                sys_msg = (self.messages[0]['content'][:200] + '...') if self.messages and 'content' in self.messages[0] else '<no-system>'
                logger.info(f"OpenAI system message: {sys_msg} (messages={len(self.messages)})", module="llm_base")
            except Exception:
                pass
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
                max_tokens=500,  # Reasonable limit for quick responses
                temperature=0.7
            )
            
            logger.info(f"🧠 LLM: Stream created, starting to iterate...")
            assistant_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    logger.info(f"🧠 LLM: Yielding content: '{content}'")
                    
                    # Track first token time
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.debug(f"First token after: {(first_token_time - start_time)*1000:.1f}ms")
                    
                    assistant_response += content
                    total_tokens += 1
                    yield content
                    
            logger.info(f"🧠 LLM: Stream completed. Total response: '{assistant_response}'")
                    
            # Add assistant response to conversation history
            self.messages.append({"role": "assistant", "content": assistant_response})
            
            # Keep conversation history manageable (last 10 messages)
            if len(self.messages) > 11:  # system + 10 messages
                self.messages = [self.messages[0]] + self.messages[-10:]
                
            total_time = time.time() - start_time
            logger.debug(f"Response complete: {total_tokens} tokens in {total_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
            
    def generate_quick_response(self, user_input: str) -> str:
        """
        Generate a quick, non-streaming response
        
        Args:
            user_input: User's message
            
        Returns:
            str: Complete response
        """
        try:
            # Create a separate message list for quick response
            quick_messages = [
                {"role": "system", "content": self.system_prompt + " Be very brief."},
                {"role": "user", "content": user_input}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=quick_messages,
                max_tokens=50,  # Very short response
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"🧠 Error generating quick response: {e}", module="llm_base")
            return "I'm thinking..."
            
    def abort_generation(self):
        """Abort ongoing generation (OpenAI doesn't support this directly)"""
        logger.debug("Generation abort requested (not supported by OpenAI API)")
        
    def clear_history(self):
        """Clear conversation history"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.info("Conversation history cleared")
        
    def measure_latency(self) -> float:
        """Measure average response latency"""
        logger.info("Measuring latency...")
        
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'test'"}
        ]
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=test_messages,
                max_tokens=5,
                temperature=0
            )
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"Latency: {latency:.1f}ms")
            return latency
            
        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            return 0.0


if __name__ == "__main__":
    # Test the LLM module
    import sys
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Please set OPENAI_API_KEY environment variable", module="llm_test")
        sys.exit(1)
        
    llm = MinimalLLM()
    
    # Prewarm
    llm.prewarm()
    
    # Measure latency
    llm.measure_latency()
    
    # Interactive test
    logger.info("\nChat with the LLM. Type 'quit' to exit.", module="llm_test")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
                
            if not user_input:
                continue
                
            print("Assistant: ", end="", flush=True)
            
            # Stream response
            for chunk in llm.stream_response(user_input):
                print(chunk, end="", flush=True)
            print()  # New line after response
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!", module="llm_test")
            break
        except Exception as e:
            logger.error(f"Error: {e}", module="llm_test")