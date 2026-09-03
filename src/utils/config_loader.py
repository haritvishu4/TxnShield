import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        # Try relative to current project root if running from elsewhere
        candidate = Path(__file__).resolve().parents[2] / config_path
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
