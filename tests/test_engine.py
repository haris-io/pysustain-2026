"""Tests for the scheme discovery engine."""

import json
import tempfile
import unittest
from pathlib import Path

from civiclens.engine import Engine


class TestEngine(unittest.TestCase):
    def test_search_schemes_returns_relevant_records(self):
        with tempfile.TemporaryDirectory() as directory:
            schemes = [
                {"id": "health", "name": "Health Support", "description": "Medical coverage"},
                {"id": "education", "name": "Student Grant", "description": "Education funding"},
            ]
            Path(directory, "schemes.json").write_text(json.dumps(schemes), encoding="utf-8")

            results = Engine(directory).search_schemes("health")

        self.assertEqual([scheme["id"] for scheme in results], ["health"])


if __name__ == "__main__":
    unittest.main()
