"""Create practical next-step suggestions for retrieved schemes."""


class ActionPlanner:
    """Produce a consistent action record for each scheme."""

    def plan_actions(self, schemes, user_profile=None):
        """Return one actionable suggestion per scheme."""
        return [
            {
                "scheme_id": scheme.get("id"),
                "scheme_name": scheme.get("name", "Unnamed scheme"),
                "action": (
                    f"Review the eligibility requirements and application process "
                    f"for {scheme.get('name', 'this scheme')}."
                ),
            }
            for scheme in schemes
        ]
