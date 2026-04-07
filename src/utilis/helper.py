"""File contains utility functions for project."""
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_units(config_path: str | Path = "config/units.yml") -> dict:
    """Load units config from YAML (cached)."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def units(unit: str, config_path: str | Path = "config/units.yml") -> str:
    """Return the SI unit for that particular unit name."""
    cfg = load_units(config_path)
    if unit in cfg.get("canonical", {}):
        return cfg["canonical"][unit]

    error = f"Unit not found for: {unit}. Checked canonical sections in {config_path}."
    raise KeyError(error)

def get_param_info(param_name: str, config_path: str | Path = "config/parameters.yml") -> dict:
    """Get parameter info from YAML config."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if param_name in cfg.get("physical", {}):
        name = "physical"
    elif param_name in cfg.get("environmental", {}):
        name = "environmental"
    elif param_name in cfg.get("simulation", {}):
        name = "simulation"
    else:
        error = f"Parameter '{param_name}' not found in {config_path}."
        raise KeyError(error)

    # convert unit to SI unit
    param_info = cfg[name][param_name]
    param_info["unit"] = units(param_info["unit"])
    return param_info

def get_local_config() -> dict:
    """Get local config from config/local_config.yml.

    LATER USE TERRAFORM TO MANAGE THIS FILE AND MAKE SURE IT'S NOT COMMITTED TO GIT.
    """
    path = Path("config/local_config.yml")
    if not path.exists():
        error = f"Local config file not found at: {path}"
        raise FileNotFoundError(error)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
