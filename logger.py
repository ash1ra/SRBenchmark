import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def create_logger(
    log_level: str = "INFO",
    log_file_name: str = "SRBenchmark",
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger("SR_Benchmark")

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    Path("logs").mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    full_log_file_name = f"logs/{log_file_name}_{current_date}.log"

    file_handler = RotatingFileHandler(
        filename=full_log_file_name,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


logger = create_logger(log_level="INFO")
