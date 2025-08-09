# utils.py
import re
import sys
from pathlib import Path
from datetime import datetime
from colorama import init, Fore

from . import directories  # Assuming directories.py is in the same package

init(autoreset=True)  # Initialize colorama for colored output

ANSI_ESCAPE = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
TIMESTAMP_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

class Logger:
    _instance = None  # Keep a reference to active logger

    def __init__(self, logfile_path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logfile_path = logfile_path or directories.PROJECT_DIR / f"process_{timestamp}.log"
        self.terminal = sys.__stdout__
        self.logfile = open(self.logfile_path, "a")

        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        if message.strip():  # Only timestamp non-empty lines
             if not TIMESTAMP_PREFIX.match(message):
                message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

        self.terminal.write(message)
        self.logfile.write(ANSI_ESCAPE.sub('', message))  # Remove ANSI codes for log file

    def flush(self):
        # Needed for Python's `print()` buffering behavior
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        Logger._instance = None
    
    @classmethod
    def setup(cls, logfile_path=None):
        """Start logging globally"""
        if cls._instance is None:
            cls._instance = cls(logfile_path)  # constructor already sets sys.stdout/sys.stderr
        return cls._instance

    @classmethod
    def teardown(cls):
        """Stop logging globally"""
        if cls._instance:
            cls._instance.close()

def color_text(text: str, r: int, g: int, b: int) -> str:
    """Return a string wrapped in 24-bit RGB ANSI escape codes."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def _log(level, color_code, msg):
    prefix = f"{color_code}[{level}]{Fore.RESET} "
    print(prefix + msg)

def process_step(msg): _log("PROCESS", color_text("", 209, 255, 246), msg)
def info(msg): _log("INFO", Fore.CYAN, msg)
def warn(msg): _log("WARNING", Fore.YELLOW, msg)
def error(msg): _log("ERROR", Fore.RED, msg)
def success(msg): _log("SUCCESS", Fore.GREEN, msg)

def extract_base_var(path: Path, pattern: re.Pattern) -> str:
    """
    From morans_i_results_<base_var>_<YYYYMMDD_HHMM>.csv -> <base_var>
    """
    stem = path.stem  # no extension
    print(f"{stem=}")
    m = pattern.match(stem)
    if not m:
        raise ValueError(f"Unexpected filename pattern: {path.name}")
    return m.group(1)  # base_var

if __name__ == "__main__":
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")

    log_path = directories.PROJECT_DIR/ f"process_{timestamp}.log"
    # Activate logging
    logger = Logger(log_path)

    # Example prints (go to both console and log file)
    print("This goes to both console and the log file.")
    print("Useful for debugging and record-keeping.")

    info("This is some debug output")
    warn("This might be a problem")
    error("This definitely is")