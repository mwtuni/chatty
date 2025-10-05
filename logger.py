# logger.py - Simple timestamped logging utility with thread information
import datetime
import threading

# Set logger level (INFO shows info, warn, error; DEBUG shows all)
LOG_LEVEL = "INFO"  # Change to "DEBUG" for detailed debugging

def should_log(level):
    levels = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    return levels.get(level.upper(), 0) >= levels.get(LOG_LEVEL, 1)

def logger(level, message, module="app"):
    """Timestamped logger with thread information"""
    if should_log(level):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_id = threading.get_ident()
        print(f"[{timestamp}] [{module}] [thread {thread_id}] [{level.lower()}] {message}")

def info(msg, module="app"): logger("info", msg, module)
def debug(msg, module="app"): logger("debug", msg, module)
def error(msg, module="app"): logger("error", msg, module)
def warn(msg, module="app"): logger("warn", msg, module)