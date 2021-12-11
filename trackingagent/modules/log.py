import logging


# # create a file handler
# handler = logging.FileHandler("rainman.log")
# handler.setLevel(logging.INFO)
# # create a logging format
# formatter = logging.Formatter(file_logging_format)
# handler.setFormatter(formatter)

# # add the handlers to the logger
# logger.addHandler(handler)
LOG_FILE = "rainman.txt"  # g.config["log_file"]

# logger.info("save this to the log")
console_logging_format = "%(levelname)s:%(asctime)s: %(message)s"
file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

# configure logger
logging.basicConfig(level=logging.DEBUG, format=console_logging_format)
logger = logging.getLogger()
# create a file handler for output file
handler = logging.FileHandler(LOG_FILE)

# set the logging level for log file
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter(file_logging_format)
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.info("Initializing Log")


class Log:
    def __init__(self):
        msg = "Initializing log"
        # format_string = "%(levelname)s: %(asctime)s: %(message)s"
        # console_logging_format = "%(levelname)s:&amp;nbsp; %(message)s"
        # file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

        # logging.basicConfig(level=logging.DEBUG, format=console_logging_format)

        # logger = logging.getLogger
        # set different formats for logging output

    def debug(self, message):
        msg = "DEBUG: {}".format(message)
        logger.debug(msg)

    def error(self, message):
        msg = "ERROR: {}".format(message)
        self.error(msg)

    def info(self, message):
        msg = "INFO: {}".format(message)
        logger.info(msg)
