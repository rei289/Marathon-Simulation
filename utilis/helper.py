"""
This file contains utility functions for saving data to JSON and CSV formats.
"""
import json
from typing import Any


def extract_global_json(var_name: str) -> Any:
    """
    Extract a variable from globals.json.
    """
    with open('globals.json', 'r') as f:
        config = json.load(f)
    return config[var_name]