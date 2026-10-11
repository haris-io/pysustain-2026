"""Tests for scheme data loading and validation."""

import json
import tempfile
import unittest
from pathlib import Path

from civiclens.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def test_loads_scheme_records(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = [{"id": "scheme_1", "name": "Health Scheme", "description": "Coverage"}]
            Path(directory, "schemes.json").write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(DataLoader(directory).load_schemes(), expected)

    def test_rejects_non_array_data(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "schemes.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON array"):
                DataLoader(directory).load_schemes()

    def test_rejects_scheme_without_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "schemes.json").write_text('[{"name": "Missing id"}]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "id"):
                DataLoader(directory).load_schemes()


if __name__ == "__main__":
    unittest.main()
