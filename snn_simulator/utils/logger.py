import logging
import os
import datetime


def setup_logger(name: str, log_root: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Setup root logger for whole project
    :param name: "" for root logger,__name__ for specific module
    :param log_root: save path of log file
    :param level: logging level
    :return:logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # top_package_name  = __name__.split('.',1)[0]
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_root:
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_root, f"{time_str}.txt")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file,mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
