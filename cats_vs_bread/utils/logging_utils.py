import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str | None = None, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level.upper())

    return logger
