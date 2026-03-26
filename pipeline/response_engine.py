def _generate_partial_response() -> str:
    """Honeypot response — encourages attacker to reveal full strategy
    without providing any harmful information."""
    return "Let's approach this step by step. Could you clarify your intent?"


_STAGE_CONFIG = {
    "safe": {
        "decision": "allow",
        "message": None,
    },
    "probing": {
        "decision": "allow_with_monitoring",
        "message": None,
    },
    "elevated": {
        "decision": "soft_warning",
        "message": "I'm here to help, but I can't assist with certain types of requests.",
    },
    "high_risk": {
        "decision": "block_with_message",
        "message": "I cannot process this request due to safety policies.",
    },
    "critical": {
        "decision": "block_and_reset",
        "message": "Session reset due to repeated unsafe interactions.",
    },
}


class ResponseEngine:
    """Graduated response engine. Maps the final session_risk to a response
    stage and decides how the system should respond. Scales with risk
    WITHOUT clearly revealing detection at lower levels.

    Deterministic, < 1ms, never exposes internal scoring to users."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_stage(session_risk: float) -> str:
        if session_risk < 0.3:
            return "safe"
        elif session_risk < 0.5:
            return "probing"
        elif session_risk < 0.7:
            return "elevated"
        elif session_risk < 0.9:
            return "high_risk"
        return "critical"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pipeline_state: dict) -> dict:
        session_risk = pipeline_state.get("session_risk", 0.0)
        stage = self._classify_stage(session_risk)
        cfg = _STAGE_CONFIG[stage]

        # Core action output
        pipeline_state["action"] = {
            "stage": stage,
            "decision": cfg["decision"],
            "response_message": cfg["message"],
        }

        # Stage-specific side-effects
        if stage == "probing":
            pipeline_state["system_prompt_adjustment"] = "tighten_safety"

        if stage in ("elevated", "high_risk"):
            pipeline_state["honeypot_response"] = _generate_partial_response()

        if stage == "high_risk":
            pipeline_state["cooldown"] = True

        if stage == "critical":
            pipeline_state["reset_context"] = True
            pipeline_state["human_review"] = True

        return pipeline_state
