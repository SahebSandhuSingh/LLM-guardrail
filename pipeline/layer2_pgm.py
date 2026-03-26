_HIGH_RISK_PAIRS = [
    ("prompt_injection", "jailbreak"),
    ("jailbreak", "persona_hijack"),
    ("prompt_injection", "persona_hijack"),
    ("pii_extraction", "prompt_injection"),
]

_PAIR_THRESHOLD = 0.4
_MULTI_SIGNAL_THRESHOLD = 0.3
_MULTI_SIGNAL_MIN_COUNT = 3
_MULTI_BOOST_VALUE = 0.15


class PGMLayer:
    """PGM correlation reasoning layer. Amplifies co-occurring threats
    instead of treating category scores independently.

    Pure deterministic logic — no ML models. Executes under 1ms."""

    def process(self, pipeline_state: dict) -> dict:
        scores = pipeline_state.get("category_scores", {})
        norm_signal = pipeline_state.get("normalization_risk_signal", 0.0)
        triggered: list[str] = []

        # 1. Base risk — highest single-category score
        base_risk = max(scores.values()) if scores else 0.0

        # 2. Pairwise amplification
        pair_boost = 0.0
        for cat_a, cat_b in _HIGH_RISK_PAIRS:
            score_a = scores.get(cat_a, 0.0)
            score_b = scores.get(cat_b, 0.0)
            if score_a > _PAIR_THRESHOLD and score_b > _PAIR_THRESHOLD:
                pair_boost += score_a * score_b
                triggered.append(f"{cat_a} + {cat_b} correlation")

        # 3. Multi-signal boost
        above_threshold = sum(
            1 for v in scores.values() if v > _MULTI_SIGNAL_THRESHOLD
        )
        if above_threshold >= _MULTI_SIGNAL_MIN_COUNT:
            multi_boost = _MULTI_BOOST_VALUE
            triggered.append("multi-signal escalation")
        else:
            multi_boost = 0.0

        # 4. Normalization / obfuscation signal integration
        if norm_signal > 0.5:
            obfuscation_boost = 0.2
            triggered.append("high obfuscation detected")
        else:
            obfuscation_boost = norm_signal * 0.2

        # 5. Final clamped score
        pgm_risk = base_risk + pair_boost + multi_boost + obfuscation_boost
        pgm_risk = min(pgm_risk, 1.0)

        # Write outputs — do NOT overwrite existing keys
        pipeline_state["pgm_risk"] = round(pgm_risk, 4)
        pipeline_state["pgm_breakdown"] = {
            "base_risk": round(base_risk, 4),
            "pair_boost": round(pair_boost, 4),
            "multi_boost": round(multi_boost, 4),
            "obfuscation_boost": round(obfuscation_boost, 4),
        }
        pipeline_state["triggered_rules"] = triggered
        return pipeline_state
