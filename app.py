# Main application entry point
# PySustain 2026 Hackathon - VIT SENSE

from civiclens.chat_service import ChatService
from civiclens.engine import Engine
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    """Initialize and run the application."""
    engine = Engine()
    chat_service = ChatService(engine)
    chat_service.start()

if __name__ == "__main__":
    main()
