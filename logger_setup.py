import logging
import sys


def setup_loggers():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Named loggers share the root handlers (can be extended with FileHandlers)
    loggers = {'root': root_logger, 'text': root_logger, 'tools': root_logger}
    return loggers


_loggers = None


def get_logger(name='root'):
    global _loggers
    if _loggers is None:
        _loggers = setup_loggers()
    return _loggers.get(name, _loggers['root'])