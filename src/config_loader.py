"""config_loader.py — loads settings.yaml into a Config object."""
import os
import yaml
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env

class ConfigLoader:
    config = {}
    config_path = None
    env_vars = {}
    load_env_vars = True
    env_file = ".env"
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "settings.yaml")
        self.config_path = os.path.abspath(config_path)
        self.load()

    def get(self, key, default=None):
        if key in self.env_vars:
            return self.env_vars[key]
        return self.config.get(key, default)
    def get_all(self):
        return self.config
            
    def load(self):
        if self.load_env_vars:
            self.env_vars = {k: v for k, v in os.environ.items() if k.isupper()}
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        return self.config


