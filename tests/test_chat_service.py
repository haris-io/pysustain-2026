"""
Tests for chat_service module
"""

import unittest
import tempfile
import json
import os
from civiclens.engine import Engine
from civiclens.chat_service import ChatService


class TestChatService(unittest.TestCase):
    """Test cases for ChatService class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_schemes = [
            {
                "id": "scheme_001",
                "name": "Health Scheme",
                "description": "Health insurance"
            }
        ]
        
        schemes_file = os.path.join(self.temp_dir, "schemes.json")
        with open(schemes_file, 'w') as f:
            json.dump(self.test_schemes, f)
        
        self.engine = Engine(self.temp_dir)
        self.chat_service = ChatService(self.engine, use_mock_llm=True)

    def test_chat_service_initialization(self):
        """Test chat service initialization."""
        self.assertIsNotNone(self.chat_service)

    def test_process_message(self):
        """Test message processing."""
        response = self.chat_service.process_message("What health schemes are available?")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_conversation_history(self):
        """Test conversation history tracking."""
        self.chat_service.process_message("Hello")
        history = self.chat_service.get_conversation_history()
        
        self.assertEqual(len(history), 2)  # User + Assistant
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")


if __name__ == '__main__':
    unittest.main()
