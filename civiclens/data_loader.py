"""Load and validate scheme data from JSON."""

import json
from pathlib import Path


class DataLoader:
    """Read scheme records from a data directory."""

    def __init__(self, data_dir=None):
        project_root = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir is not None else project_root / "data"
        self.schemes_file = self.data_dir / "schemes.json"

    def load_schemes(self):
        """Return validated scheme records from ``schemes.json``."""
        with self.schemes_file.open(encoding="utf-8") as file:
            schemes = json.load(file)

        if not isinstance(schemes, list):
            raise ValueError("schemes.json must contain a JSON array")

        for index, scheme in enumerate(schemes):
            if not isinstance(scheme, dict):
                raise ValueError(f"Scheme at index {index} must be a JSON object")
            if not isinstance(scheme.get("id"), str) or not scheme["id"].strip():
                raise ValueError(f"Scheme at index {index} must have a non-empty string id")
            if not isinstance(scheme.get("name"), str) or not scheme["name"].strip():
                raise ValueError(f"Scheme at index {index} must have a non-empty string name")
            if "description" in scheme and not isinstance(scheme["description"], str):
                raise ValueError(f"Scheme at index {index} description must be a string")

        return schemes
