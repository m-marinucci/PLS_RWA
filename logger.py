import logging
import sys

def setup_logger(debug_mode=False):
    """Setup logger with appropriate level and format."""
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # Create logger
    logger = logging.getLogger('simulation')
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create console handler with appropriate format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # Create formatters
    if debug_mode:
        formatter = logging.Formatter('%(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
