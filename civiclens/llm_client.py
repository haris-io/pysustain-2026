"""Compose user-facing responses from local scheme data."""


class LLMClient:
    """Format scheme search results into a concise response.

    This local implementation does not send user data to an external service.
    It provides a deterministic baseline for the application and a seam for a
    future hosted language-model integration.
    """

    def __init__(self, use_mock=False):
        self.use_mock = use_mock

    def generate_response(self, message, schemes, actions):
        """Return a readable response based on the retrieved schemes."""
        if not schemes:
            return "I couldn't find any schemes matching that request in the available data."

        lines = ["Here are the schemes that may be relevant:"]
        action_by_id = {action["scheme_id"]: action for action in actions}
        for scheme in schemes:
            name = scheme.get("name", "Unnamed scheme")
            description = (
                scheme.get("description") or scheme.get("summary") or scheme.get("benefit_summary", "")
            ).strip()
            scheme_id = scheme.get("scheme_id", scheme.get("id"))
            lines.append(f"- {name}" + (f": {description}" if description else ""))
            action = action_by_id.get(scheme_id)
            if action:
                lines.append(f"  Next step: {action['action']}")

        lines.append("Check the official scheme source for current eligibility and application details.")
        return "\n".join(lines)
