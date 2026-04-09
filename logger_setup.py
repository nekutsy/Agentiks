import logging
import sys
from config import LOG_GENERATED_TEXT, LOG_GENERATED_TOOLS, LOG_CURRENT_TOOL

def setup_loggers():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    text_handler = logging.FileHandler(LOG_GENERATED_TEXT, encoding='utf-8')
    text_handler.setFormatter(formatter)
    text_logger = logging.getLogger('text')
    text_logger.setLevel(logging.INFO)
    text_logger.addHandler(text_handler)
    text_logger.propagate = False

    tools_handler = logging.FileHandler(LOG_GENERATED_TOOLS, encoding='utf-8')
    tools_handler.setFormatter(formatter)
    tools_logger = logging.getLogger('tools')
    tools_logger.setLevel(logging.INFO)
    tools_logger.addHandler(tools_handler)
    tools_logger.propagate = False

    current_tool_handler = logging.FileHandler(LOG_CURRENT_TOOL, mode='w', encoding='utf-8')
    current_tool_handler.setFormatter(formatter)
    current_tool_logger = logging.getLogger('current_tool')
    current_tool_logger.setLevel(logging.INFO)
    current_tool_logger.addHandler(current_tool_handler)
    current_tool_logger.propagate = False

    return {
        'root': root_logger,
        'text': text_logger,
        'tools': tools_logger,
        'current_tool': current_tool_logger
    }

loggers = None

def get_logger(name='root'):
    global loggers
    if loggers is None:
        loggers = setup_loggers()
    return loggers.get(name, loggers['root'])