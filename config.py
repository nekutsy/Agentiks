import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_SESSIONS = os.path.join(BASE_DIR, "sessions.json")
LOG_CURRENT_INPUT = os.path.join(BASE_DIR, "current_input.txt")

EXEC_TEMP_DIR = BASE_DIR

TEMP_PYTHON_DIR = os.path.join(BASE_DIR, "temp_python")

MODEL_NAME = "gemma4:e4b "

STREAM_MESSAGES = True
LOG_THINKING = True
THINKING_HISTORY_MODE = "all"  # "none", "all", "last"

PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPTS_DIR, "system_prompt.txt")
USER_FIRST_MESSAGE_FILE = os.path.join(PROMPTS_DIR, "user_first_message.txt")

MAX_SESSION_NUM = 1
CONTEXT_LIMIT_TOKENS = 1024*128

OLLAMA_OPTIONS = {
    "num_predict": 1024*32,
}

MAX_HISTORY_MESSAGES = 40
MAX_TOOL_RESULT_LENGTH = 20_000

OLLAMA_NUM_CTX = 32768
OLLAMA_NUM_BATCH = 2048
OLLAMA_NUM_GPU = -1
OLLAMA_NUM_THREAD = 0
OLLAMA_FLASH_ATTENTION = True
OLLAMA_KEEP_ALIVE = "24h"