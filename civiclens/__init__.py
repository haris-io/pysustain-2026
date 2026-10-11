"""CivicLens: an LLM-assisted public-benefits navigator for students in Tamil Nadu.

The backend is split into small modules with a strict one-way dependency order
(no module imports a module listed after it):

    data_loader -> engine -> retriever / action_planner -> llm_client -> chat_service

* ``data_loader``   loads and validates ``data/schemes.json``.
* ``engine``        deterministic eligibility evaluation (no LLM involved).
* ``retriever``     lightweight local term-based retrieval (no network).
* ``action_planner`` deterministic, ordered action plans.
* ``llm_client``    OpenAI Responses API calls: profile extraction + explanations.
* ``chat_service``  the end-to-end pipeline used by the Streamlit UI.

Importing this package is intentionally cheap: it does not import the
submodules (and therefore does not import the OpenAI SDK).
"""

__version__ = "1.0.0"

__all__ = [
    "data_loader",
    "engine",
    "retriever",
    "action_planner",
    "llm_client",
    "chat_service",
]
