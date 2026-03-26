_CATEGORY_DISPLAY = {
    "prompt_injection": "prompt injection",
    "jailbreak": "jailbreak",
    "pii_extraction": "PII extraction",
    "persona_hijack": "persona hijack",
}


class ExplainabilityLayer:
    """Explainability engine. Converts all upstream layer outputs into a
    structured, human-readable explanation for debugging, auditing, and demos.

    Pure formatting + logic — no ML, no external calls. < 1ms."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level(session_risk: float) -> str:
        if session_risk < 0.3:
            return "low"
        elif session_risk < 0.6:
            return "medium"
        elif session_risk < 0.85:
            return "high"
        return "critical"

    @staticmethod
    def _primary_threat(category_scores: dict) -> str:
        if not category_scores:
            return "none"
        return max(category_scores, key=lambda k: category_scores[k])

    @staticmethod
    def _active_threats(category_scores: dict, threshold: float = 0.5) -> list[str]:
        return [cat for cat, score in category_scores.items() if score > threshold]

    @staticmethod
    def _contributing_factors(
        norm_signal: float,
        triggered_rules: list,
        session_flags: list,
        category_scores: dict,
    ) -> list[str]:
        factors: list[str] = []

        if norm_signal > 0.4:
            factors.append("input obfuscation detected")

        for rule in triggered_rules:
            factors.append(rule)

        for flag in session_flags:
            factors.append(flag)

        multi_count = sum(1 for v in category_scores.values() if v > 0.4)
        if multi_count >= 2:
            factors.append("multiple threat signals detected")

        return factors

    @staticmethod
    def _build_summary(
        risk_level: str,
        primary_threat: str,
        active_threats: list[str],
        factors: list[str],
        session_risk: float,
    ) -> str:
        if risk_level == "low":
            return "Input appears safe. No significant threats detected."

        display = _CATEGORY_DISPLAY.get(primary_threat, primary_threat)

        parts: list[str] = []
        if active_threats:
            threat_names = [_CATEGORY_DISPLAY.get(t, t) for t in active_threats]
            if len(threat_names) == 1:
                parts.append(f"This input shows signs of {threat_names[0]}.")
            else:
                joined = ", ".join(threat_names[:-1]) + " and " + threat_names[-1]
                parts.append(f"This input shows signs of {joined}.")
        else:
            parts.append(f"The primary concern is {display}.")

        if any("obfuscation" in f for f in factors):
            parts.append("The input contained obfuscated characters, increasing suspicion.")

        if any("correlation" in f for f in factors):
            parts.append("Multiple signals combined to amplify the risk score.")

        if any("session" in f.lower() or "previous" in f.lower() or "trending" in f.lower() for f in factors):
            parts.append("Prior session behavior contributed to elevated risk.")

        parts.append(f"Overall risk level: {risk_level} ({session_risk:.2f}).")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pipeline_state: dict) -> dict:
        # Read upstream outputs — gracefully handle missing fields
        category_scores = pipeline_state.get("category_scores", {})
        norm_signal = pipeline_state.get("normalization_risk_signal", 0.0)
        pgm_breakdown = pipeline_state.get("pgm_breakdown", {})
        triggered_rules = pipeline_state.get("triggered_rules", [])
        session_risk = pipeline_state.get("session_risk", 0.0)
        session_breakdown = pipeline_state.get("session_breakdown", {})
        session_flags = pipeline_state.get("session_flags", [])

        risk_level = self._risk_level(session_risk)
        primary_threat = self._primary_threat(category_scores)
        active_threats = self._active_threats(category_scores)
        factors = self._contributing_factors(
            norm_signal, triggered_rules, session_flags, category_scores,
        )
        summary = self._build_summary(
            risk_level, primary_threat, active_threats, factors, session_risk,
        )

        pipeline_state["explanation"] = {
            "risk_level": risk_level,
            "primary_threat": primary_threat,
            "active_threats": active_threats,
            "contributing_factors": factors,
            "summary": summary,
        }

        pipeline_state["debug_info"] = {
            "category_scores": dict(category_scores),
            "pgm_breakdown": dict(pgm_breakdown),
            "session_breakdown": dict(session_breakdown),
        }

        return pipeline_state
