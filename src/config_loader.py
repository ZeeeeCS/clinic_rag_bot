"""config_loader.py — loads settings.yaml into a Config object."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
for env_path in (ROOT_DIR / ".env", ROOT_DIR / "env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

# Load environment variables from .env or env

class ConfigLoader:

    env_file = ".env"
    def __init__(self, config_path=None):

        if config_path is None:
            config_path = os.path.join(os.getcwd(), "config", "settings.yaml")
        self.config_path = os.path.abspath(config_path)
        self.config = {}
        self.load()

    def get(self, key, default=Optional[Any] or List[Any] or Dict[Any, Any]):
        """Retrieve a configuration value by key, with an optional default."""
        if not self.config:
            self.load()
        
            return self.config.get(key, default)

        return self.config.get(key, default)

    def get_model_setting(self, name, default=None):
        models = self.config.get("models", {})
        if not isinstance(models, dict):
            return default

        value = models.get(name)
        if isinstance(value, dict):
            provider = value.get("provider") or "groq"
            return {
                "model": value.get("model") or value.get("name") or "",
                "provider": provider,
            }

        if isinstance(value, str):
            provider = models.get(f"{name}_provider") or "groq"
            return {"model": value, "provider": provider}

        return default

    def get_all(self):
        return self.config
            
    def load(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file not found at {self.config_path}")
            return {}
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}
        return self.config


