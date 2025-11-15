"""Centralized logging utilities for Model Chat CLI."""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "model_chat_cli", log_to_file: bool = True) -> logging.Logger:
    """Set up and configure logger.

    Args:
        name: Logger name
        log_to_file: Whether to log to file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler - DEBUG and above
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stress_test_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def log_request_error(logger: logging.Logger, request_id: int, error: Exception, context: dict = None):
    """Log request error with context.

    Args:
        logger: Logger instance
        request_id: Request identifier
        error: Exception that occurred
        context: Additional context dict
    """
    error_type = type(error).__name__
    error_msg = str(error)

    log_msg = f"Request #{request_id} failed: {error_type} - {error_msg}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        log_msg += f" | Context: {context_str}"

    logger.error(log_msg)


def log_vllm_error(logger: logging.Logger, request_id: int, response_text: str):
    """Log VLLM-specific error from server response.

    Args:
        logger: Logger instance
        request_id: Request identifier
        response_text: Response text from server
    """
    logger.error(f"Request #{request_id} VLLM error: {response_text}")


def log_test_summary(logger: logging.Logger, mode: str, stats: dict):
    """Log test summary statistics.

    Args:
        logger: Logger instance
        mode: Test mode name
        stats: Statistics dictionary
    """
    logger.info(f"Test Mode: {mode}")
    logger.info(f"Total Requests: {stats.get('total', 0)}")
    logger.info(f"Successful: {stats.get('success', 0)}")
    logger.info(f"Failed: {stats.get('failed', 0)}")
    logger.info(f"Average Response Time: {stats.get('avg_response_time', 0):.2f}s")
    logger.info(f"Average Tokens/sec: {stats.get('avg_tps', 0):.2f}")
