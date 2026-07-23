"""logger.py — structured logging setup (console + file)."""
import logging
import os
from config_loader import ConfigLoader

class Logger:
    def __init__(self):
        self.config = ConfigLoader()
        # Use a named logger for the bot
        self.logger = logging.getLogger("clinic_rag_bot")
        
        # Avoid duplicate handlers if Logger is instantiated multiple times
        if not self.logger.handlers:
            log_config = self.config.get("logging", {})
            log_level_str = log_config.get("level", "INFO")
            log_level = getattr(logging, log_level_str.upper(), logging.INFO)
            self.logger.setLevel(log_level)
            
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            
            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File Handler
            log_file = log_config.get("log_file", "logs/app.log")
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info("Logger initialized")
