"""Coordinate scheme data loading and retrieval."""

from .data_loader import DataLoader
from .retriever import Retriever


class Engine:
    """Application-level interface for looking up schemes."""

    def __init__(self, data_dir=None):
        self.data_loader = DataLoader(data_dir)
        self.retriever = Retriever(self.data_loader)

    def search_schemes(self, query, user_profile=None):
        """Find schemes relevant to a user's query."""
        return self.retriever.retrieve(query, user_profile)
