import os
import logging
import datetime
from typing import Any, Self

class MyLogger:
    _instance = None
    _logger: logging.Logger = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = logging.getLogger("FaceDetection")
            cls._instance._logger.setLevel(logging.INFO)
            cls._instance._logger.propagate = False  # tránh log trùng
        return cls._instance

    def setup(self, log_level=logging.INFO, log_to_console=True, log_folder='logs'):
        if hasattr(self, 'handler_configured') and self.handler_configured:
            return

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, f"log_{now}.log")

        handlers = [logging.FileHandler(log_file, mode='w')]
        if log_to_console:
            handlers.append(logging.StreamHandler())

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        for handler in handlers:
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._logger.setLevel(log_level)
        self.handler_configured = True

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)
