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

    log_config = {
        'text': (LOG_GENERATED_TEXT, 'a'),
        'tools': (LOG_GENERATED_TOOLS, 'a'),
        'current_tool': (LOG_CURRENT_TOOL, 'w')
    }

    loggers = {'root': root_logger}
    for name, (filename, mode) in log_config.items():
        handler = logging.FileHandler(filename, mode, encoding='utf-8')
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        loggers[name] = logger

    return loggers

loggers = None

def get_logger(name='root'):
    global loggers
    if loggers is None:
        loggers = setup_loggers()
    return loggers.get(name, loggers['root'])