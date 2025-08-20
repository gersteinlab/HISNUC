import logging

# Create a global logger instance
logger = logging.getLogger()

def setup_logger(run_path):
    """
    Set up a logger that writes INFO-level messages to both a file and the console.

    Parameters:
    - run_path (Path): Directory path where the log file 'output.log' will be stored.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Set the global logging level to DEBUG (captures all levels)
    logger.setLevel(logging.DEBUG)

    # File handler: logs INFO and above to 'output.log'
    file_handler = logging.FileHandler(run_path / "output.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    # Console handler: logs INFO and above to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))  # Add formatting for consistency
    logger.addHandler(console_handler)
