"""Coordinate scheme data loading and retrieval."""

from pathlib import Path

from .data_loader import DEFAULT_SCHEMES_PATH, load_schemes
from .retriever import Retriever


class Engine:
    """Application-level interface for looking up schemes."""

    def __init__(self, data_dir=None):
        if data_dir is None:
            self.schemes_path = DEFAULT_SCHEMES_PATH
        else:
            path = Path(data_dir)
            self.schemes_path = path / "schemes.json" if path.is_dir() or not path.suffix else path
        self.retriever = Retriever(self)

    def load_schemes(self):
        """Load the configured, validated scheme catalogue."""
        return load_schemes(self.schemes_path)

    def search_schemes(self, query, user_profile=None):
        """Find schemes relevant to a user's query."""
        return self.retriever.retrieve(query, user_profile)
