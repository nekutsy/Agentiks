import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_SESSIONS = os.path.join(BASE_DIR, "sessions.json")
LOG_GENERATED_TEXT = os.path.join(BASE_DIR, "generated_text.log")
LOG_GENERATED_TOOLS = os.path.join(BASE_DIR, "generated_tools.log")
LOG_CURRENT_TOOL = os.path.join(BASE_DIR, "current_tool_execution.txt")
LOG_CURRENT_INPUT = os.path.join(BASE_DIR, "current_input.txt")

EXEC_TEMP_DIR = os.path.join(BASE_DIR, "current.py")

TEMP_PYTHON_DIR = os.path.join(BASE_DIR, "temp_python")

MODEL_NAME = "gemma4:e4b"

OLLAMA_OPTIONS = {
    "num_predict": 1024*32,
}
STREAM_MESSAGES = True
LOG_THINKING = True
THINKING_HISTORY_MODE = "all"  # "none", "all", "last"

PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPTS_DIR, "system_prompt.txt")
USER_FIRST_MESSAGE_FILE = os.path.join(PROMPTS_DIR, "user_first_message.txt")

MAX_SESSION_NUM = 1
CONTEXT_LIMIT_TOKENS = 1024*128

IS_RUN_FILE = os.path.join(BASE_DIR, "is_run")