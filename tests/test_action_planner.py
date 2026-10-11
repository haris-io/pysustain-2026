"""
Tests for action_planner module
"""

import unittest
from civiclens.action_planner import ActionPlanner


class TestActionPlanner(unittest.TestCase):
    """Test cases for ActionPlanner class."""

    def setUp(self):
        """Set up test fixtures."""
        self.planner = ActionPlanner()
        self.test_schemes = [
            {
                "id": "scheme_001",
                "name": "Health Scheme",
                "description": "Health insurance"
            },
            {
                "id": "scheme_002",
                "name": "Education Scheme",
                "description": "Education grants"
            }
        ]

    def test_plan_actions(self):
        """Test action planning."""
        actions = self.planner.plan_actions(self.test_schemes)
        
        self.assertEqual(len(actions), 2)
        self.assertIn("scheme_id", actions[0])
        self.assertIn("scheme_name", actions[0])
        self.assertIn("action", actions[0])

    def test_plan_for_empty_schemes(self):
        """Test planning with empty schemes list."""
        actions = self.planner.plan_actions([])
        
        self.assertEqual(len(actions), 0)

    def test_plan_with_user_profile(self):
        """Test planning with user profile."""
        user_profile = {
            "age": 25,
            "income": 200000
        }
        
        actions = self.planner.plan_actions(self.test_schemes, user_profile)
        
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)


if __name__ == '__main__':
    unittest.main()
