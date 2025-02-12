import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {} 