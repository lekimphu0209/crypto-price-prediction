"""
Config loader - reads YAML configs from the top-level configs/ directory.

Pipeline-level defaults live in configs/default.yaml; per-model hyperparams
live in configs/models/<name>.yaml. YAML is optional - if pyyaml or the file
is missing, safe built-in defaults are returned so the system never breaks.
"""
import os
from typing import Any, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(_PROJECT_ROOT, "configs")

_DEFAULTS: Dict[str, Any] = {
    "seq_len": 20,
    "horizon": 7,
    "train_days": 1000,
    "extractor": {"name": "technical", "params": {"feature_set": "A"}},
}


def load_config(name: str = "default") -> Dict[str, Any]:
    """
    Load configs/<name>.yaml merged over built-in defaults.

    Args:
        name: config file stem (without .yaml)

    Returns:
        dict of settings; missing file/keys fall back to defaults.
    """
    cfg = dict(_DEFAULTS)
    path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
    if os.path.exists(path):
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            cfg.update(loaded)
        except ImportError:
            print("[config] pyyaml not installed - using built-in defaults")
        except Exception as e:
            print(f"[config] failed to load {path}: {e} - using defaults")
    return cfg


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load per-model hyperparams from configs/models/<slug>.yaml.

    Returns {} when no config exists (model uses its own defaults).
    """
    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(CONFIGS_DIR, "models", f"{slug}.yaml")
    if not os.path.exists(path):
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[config] failed to load {path}: {e}")
        return {}
