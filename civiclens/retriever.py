"""Keyword-based retrieval of relevant schemes."""

import re


class Retriever:
    """Rank scheme records by how well their text matches a query."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def retrieve(self, query, user_profile=None):
        """Return matching schemes, ordered by descending text relevance."""
        schemes = self.data_loader.load_schemes()
        terms = set(re.findall(r"\w+", str(query).casefold()))
        if not terms:
            return schemes

        ranked = []
        for index, scheme in enumerate(schemes):
            searchable_text = " ".join(
                str(value)
                for key, value in scheme.items()
                if key not in {"id"} and isinstance(value, (str, int, float))
            ).casefold()
            score = sum(len(re.findall(r"\b" + re.escape(term) + r"\b", searchable_text)) for term in terms)
            if score:
                ranked.append((score, index, scheme))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [scheme for _, _, scheme in ranked]
