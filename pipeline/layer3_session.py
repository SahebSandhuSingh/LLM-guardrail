import threading
import time


_HISTORY_LIMIT = 10
_SMOOTHING_PREV = 0.7
_SMOOTHING_CURR = 0.3


class SessionRiskLayer:
    """Session risk scorer. Tracks and accumulates risk across an entire
    conversation session using exponential smoothing and escalation rules.

    State is kept in-memory and persists while the server is running."""

    def __init__(self) -> None:
        self.session_store: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, session_id: str) -> dict:
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "risk_history": [],
                "rolling_risk": 0.0,
                "last_updated": time.time(),
            }
        return self.session_store[session_id]

    @staticmethod
    def _compute_escalation(history: list[float]) -> tuple[float, list[str]]:
        boost = 0.0
        flags: list[str] = []

        # Rule 1: any previous risk > 0.8
        if any(r > 0.8 for r in history[:-1]):
            boost = max(boost, 0.15)
            flags.append("high previous risk detected")

        # Rule 2: last 3 messages avg > 0.6
        if len(history) >= 3:
            last3 = history[-3:]
            if sum(last3) / 3 >= 0.6:
                boost = max(boost, 0.2)
                flags.append("consistent attack pattern")

        # Rule 3: risk trending upward (last 3 strictly increasing)
        if len(history) >= 3:
            last3 = history[-3:]
            if last3[0] < last3[1] < last3[2]:
                boost += 0.1
                flags.append("risk trending upward")

        return boost, flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pipeline_state: dict) -> dict:
        session_id = pipeline_state.get("session_id", "unknown")
        pgm_risk = pipeline_state.get("pgm_risk", 0.0)

        with self._lock:
            session = self._get_or_create(session_id)
            history = session["risk_history"]
            prev_rolling = session["rolling_risk"]

            # Append current risk and trim to sliding window
            history.append(pgm_risk)
            if len(history) > _HISTORY_LIMIT:
                history[:] = history[-_HISTORY_LIMIT:]

            # Exponential smoothing
            if len(history) == 1:
                rolling_risk = pgm_risk
            else:
                rolling_risk = (prev_rolling * _SMOOTHING_PREV) + (pgm_risk * _SMOOTHING_CURR)

            # Escalation
            escalation_boost, flags = self._compute_escalation(history)

            # Final clamped score
            session_risk = min(rolling_risk + escalation_boost, 1.0)

            # Persist
            session["rolling_risk"] = rolling_risk
            session["last_updated"] = time.time()

        # Write to pipeline_state
        pipeline_state["session_risk"] = round(session_risk, 4)
        pipeline_state["session_breakdown"] = {
            "rolling_risk": round(rolling_risk, 4),
            "escalation_boost": round(escalation_boost, 4),
            "history_length": len(history),
        }
        pipeline_state["session_flags"] = flags
        pipeline_state["session_risk_history"] = list(history)
        return pipeline_state
