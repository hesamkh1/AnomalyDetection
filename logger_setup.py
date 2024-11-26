import logging
import os
from datetime import datetime

def setup_logger(output_dir):
    # Get the current date and time for unique log file naming
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'log_{current_time}.txt'  # Example: log_2024-10-18_12-30-45.txt

    # Create or retrieve the logger object
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)  # Set the logging level
    
    # Avoid adding duplicate handlers by clearing existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler to write to a time-stamped log file
    log_file_path = os.path.join(output_dir, log_filename)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # Create a console handler to print to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the format for both file and console
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
