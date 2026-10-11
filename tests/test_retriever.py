"""
Tests for retriever module
"""

import unittest
import tempfile
import json
import os
from civiclens.data_loader import DataLoader
from civiclens.retriever import Retriever


class TestRetriever(unittest.TestCase):
    """Test cases for Retriever class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_schemes = [
            {
                "id": "scheme_001",
                "name": "Health Insurance Scheme",
                "description": "Provides health insurance coverage"
            },
            {
                "id": "scheme_002",
                "name": "Education Grant",
                "description": "Educational support for students"
            }
        ]
        
        schemes_file = os.path.join(self.temp_dir, "schemes.json")
        with open(schemes_file, 'w') as f:
            json.dump(self.test_schemes, f)
        
        self.data_loader = DataLoader(self.temp_dir)
        self.retriever = Retriever(self.data_loader)

    def test_retrieve_schemes(self):
        """Test retrieving schemes."""
        results = self.retriever.retrieve("health")
        
        self.assertIsInstance(results, list)

    def test_retrieve_with_profile(self):
        """Test retrieving with user profile."""
        user_profile = {
            "age": 25,
            "income": 200000
        }
        
        results = self.retriever.retrieve("health", user_profile)
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    unittest.main()
