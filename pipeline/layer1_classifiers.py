import time
import re


# ---------------------------------------------------------------------------
# Keyword-based fallback scorer
# ---------------------------------------------------------------------------

_KEYWORD_PATTERNS: dict[str, list[str]] = {
    "prompt_injection": [
        "ignore previous", "ignore instructions", "disregard",
        "override", "system prompt", "forget your",
        "ignore all", "ignore the above", "do not follow",
        "new instructions", "bypass", "ignore prior",
        "tell me everything", "reveal your instructions",
        "repeat the above", "what were your instructions",
        "ignore your", "forget all", "drop all",
    ],
    "jailbreak": [
        "pretend you are", "act as if", "you are now", "jailbreak",
        "no restrictions", "without limits", "dan", "do anything now",
        "developer mode", "unrestricted mode", "ignore safety",
        "no rules", "break free", "unlocked mode",
        "answer freely", "without any restrictions",
        "no boundaries", "bypass filter", "no limitations",
    ],
    "pii_extraction": [
        "what is your", "tell me your", "reveal", "show me user",
        "access database", "list all users", "give me the password",
        "show credentials", "personal information", "social security",
        "credit card", "account balance", "private data", "email address",
        "what is the account", "user data", "phone number",
        "show me the", "list all accounts", "home address",
    ],
    "persona_hijack": [
        "you are now a", "new persona", "different ai",
        "unrestricted ai", "pretend to be", "roleplay as",
        "act as a", "become a", "switch persona", "adopt the role",
        "from now on you are", "your new identity",
        "pretend you are", "you are an ai with",
        "behave as", "take on the role", "new character",
    ],
}

# Pre-compile patterns for speed
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    cat: [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
    for cat, keywords in _KEYWORD_PATTERNS.items()
}


def _keyword_score(text: str, category: str) -> float:
    """Score text against keyword patterns for a category.
    Returns 0.0–1.0 based on number and strength of matches."""
    if not text:
        return 0.0
    patterns = _COMPILED_PATTERNS.get(category, [])
    if not patterns:
        return 0.0
    matches = sum(1 for p in patterns if p.search(text))
    if matches == 0:
        return 0.0
    # Scale: 1 match → 0.55, 2 → 0.7, 3+ → 0.85+
    score = min(0.55 + (matches - 1) * 0.15, 1.0)
    return round(score, 4)


def _keyword_fallback(text: str) -> dict[str, float]:
    """Run all categories through keyword matching."""
    return {cat: _keyword_score(text, cat) for cat in _KEYWORD_PATTERNS}


# ---------------------------------------------------------------------------
# Main classifier layer
# ---------------------------------------------------------------------------

_CATEGORIES = ["prompt_injection", "jailbreak", "pii_extraction", "persona_hijack"]

_CATEGORY_LABELS = {
    "prompt_injection": "ignoring or overriding system instructions",
    "jailbreak": "bypassing safety rules and restrictions",
    "pii_extraction": "extracting personal or private information",
    "persona_hijack": "adopting an unrestricted AI persona",
}


class Layer1Classifiers:
    """Per-category classifiers using HuggingFace zero-shot classification
    with cross-encoder/nli-distilroberta-base (~330 MB). Falls back to
    keyword scoring if the model cannot be loaded or is too slow."""

    def __init__(self) -> None:
        self._classifier = None
        self._use_fallback = False
        self._labels = [_CATEGORY_LABELS[c] for c in _CATEGORIES]
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-distilroberta-base",
                device=-1,  # CPU
            )
            # Warm-up probe: if a single call exceeds 500ms, switch to keywords
            t0 = time.perf_counter()
            self._classifier(
                "test", candidate_labels=["safe"], multi_label=True,
            )
            probe_ms = (time.perf_counter() - t0) * 1000
            if probe_ms > 500:
                self._classifier = None
                self._use_fallback = True
        except Exception:
            self._use_fallback = True

    def _classify_zero_shot(self, text: str) -> dict[str, float]:
        """Run zero-shot classification and map label descriptions back to
        category keys."""
        result = self._classifier(
            text,
            candidate_labels=self._labels,
            multi_label=True,
        )
        label_to_score = dict(zip(result["labels"], result["scores"]))
        scores: dict[str, float] = {}
        for cat, label in zip(_CATEGORIES, self._labels):
            scores[cat] = round(label_to_score.get(label, 0.0), 4)
        return scores

    def _blend_scores(self, text: str) -> dict[str, float]:
        """Combine keyword + model scores via max to get the best of both."""
        kw = _keyword_fallback(text)
        try:
            zs = self._classify_zero_shot(text)
        except Exception:
            return kw
        return {c: round(max(kw[c], zs[c]), 4) for c in _CATEGORIES}

    def process(self, pipeline_state: dict) -> dict:
        text = pipeline_state.get("normalized_text", "")

        if not text or not isinstance(text, str) or text.strip() == "":
            pipeline_state["category_scores"] = {c: 0.0 for c in _CATEGORIES}
            pipeline_state["classifier_latency_ms"] = 0.0
            return pipeline_state

        start = time.perf_counter()

        if self._use_fallback or self._classifier is None:
            scores = _keyword_fallback(text)
        else:
            scores = self._blend_scores(text)

        elapsed_ms = (time.perf_counter() - start) * 1000

        pipeline_state["category_scores"] = scores
        pipeline_state["classifier_latency_ms"] = round(elapsed_ms, 2)
        return pipeline_state
