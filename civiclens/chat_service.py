"""Conversation service for scheme discovery."""

from .action_planner import ActionPlanner
from .llm_client import LLMClient


class ChatService:
    """Connect user messages to scheme retrieval and response formatting."""

    def __init__(self, engine, use_mock_llm=False):
        self.engine = engine
        self.llm_client = LLMClient(use_mock=use_mock_llm)
        self.action_planner = ActionPlanner()
        self._conversation_history = []

    def process_message(self, message, user_profile=None):
        """Retrieve relevant schemes and append the exchange to history."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        schemes = self.engine.search_schemes(message, user_profile)
        actions = self.action_planner.plan_actions(schemes, user_profile)
        response = self.llm_client.generate_response(message, schemes, actions)
        self._conversation_history.extend(
            [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        )
        return response

    def get_conversation_history(self):
        """Return a copy of the conversation history."""
        return [dict(message) for message in self._conversation_history]

    def start(self):
        """Run an interactive terminal chat until the user exits."""
        print("CivicLens scheme assistant. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                message = input("You: ").strip()
            except EOFError:
                print()
                return
            if message.casefold() in {"exit", "quit"}:
                return
            if not message:
                continue
            print(f"CivicLens: {self.process_message(message)}")
