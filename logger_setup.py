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

    loggers = {'root': root_logger}
    return loggers

loggers = None

def get_logger(name='root'):
    global loggers
    if loggers is None:
        loggers = setup_loggers()
    return loggers.get(name, loggers['root'])