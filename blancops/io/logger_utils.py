import logging
import os
from pathlib import Path
import sys

def setup_logger(save_dir, logging_filename=None, parent_module='blancops', logging_level='debug', format=None, datefmt='%Y-%m-%d %H:%M:%S'):
    if save_dir is not None:
        assert logging_filename is not None, "Must provide logging filename if saving logs to directory. Use save_dir=None if not saving logs."
    # Create logger
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger(parent_module)
    logger.propagate = False
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if logging_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif logging_level == 'info':
        logger.setLevel(logging.INFO)
    else:
        raise NotImplementedError
    format = logging.Formatter(format, datefmt=datefmt)

    # Avoid duplicate handlers if called twice
    if logger.handlers:
        raise ValueError("Handler called twice")
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(format)
    logger.addHandler(console_handler)
    
    # Create file handler
    if save_dir is not None:
        if not os.path.exists(Path(save_dir)):
            os.mkdir(Path(save_dir))
        file_handler = logging.FileHandler(Path(save_dir) / logging_filename, mode='w')
        file_handler.setFormatter(format)
        logger.addHandler(file_handler)

    return logger
