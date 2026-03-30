"""File contains utility functions for saving data to JSON and CSV formats."""
import json
from pathlib import Path


def extract_global_json(var_name: str) -> str | float | int | bool | list | dict:
    """Extract a variable from globals.json."""
    config = json.loads(Path("globals.json").read_text(encoding="utf-8"))
    return config[var_name]

def extract_json(file_path: str) -> dict:
    """Extract data from a JSON file."""
    return json.loads(Path(file_path).read_text(encoding="utf-8"))
