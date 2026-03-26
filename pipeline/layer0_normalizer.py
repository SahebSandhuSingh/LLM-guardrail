import unicodedata
import re


# ---------------------------------------------------------------------------
# Homoglyph map: visually similar Unicode chars → ASCII equivalents
# ---------------------------------------------------------------------------
_CYRILLIC_MAP = {
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0445": "x",  # х
    "\u0410": "A",  # А
    "\u0415": "E",  # Е
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0421": "C",  # С
    "\u0425": "X",  # Х
}

_GREEK_MAP = {
    "\u03b1": "a",  # α
    "\u03bf": "o",  # ο
    "\u0391": "A",  # Α
    "\u039f": "O",  # Ο
    "\u0397": "H",  # Η (Eta)
    "\u03b5": "e",  # ε
    "\u0395": "E",  # Ε
    "\u0392": "B",  # Β
    "\u0396": "Z",  # Ζ
    "\u039a": "K",  # Κ
    "\u039c": "M",  # Μ
    "\u039d": "N",  # Ν
    "\u03a1": "P",  # Ρ
    "\u03a4": "T",  # Τ
    "\u03a5": "Y",  # Υ
    "\u03a7": "X",  # Χ
}


def _build_math_bold_italic_map() -> dict[str, str]:
    """Map Mathematical Bold, Italic, Bold-Italic, Sans-Serif (and their
    variants) of A-Z / a-z back to plain ASCII."""
    m: dict[str, str] = {}
    # (start_codepoint, ascii_start, count)
    ranges = [
        # Bold
        (0x1D400, "A", 26), (0x1D41A, "a", 26),
        # Italic
        (0x1D434, "A", 26), (0x1D44E, "a", 26),
        # Bold Italic
        (0x1D468, "A", 26), (0x1D482, "a", 26),
        # Sans-Serif
        (0x1D5A0, "A", 26), (0x1D5BA, "a", 26),
        # Sans-Serif Bold
        (0x1D5D4, "A", 26), (0x1D5EE, "a", 26),
        # Sans-Serif Italic
        (0x1D608, "A", 26), (0x1D622, "a", 26),
        # Sans-Serif Bold Italic
        (0x1D63C, "A", 26), (0x1D656, "a", 26),
        # Monospace
        (0x1D670, "A", 26), (0x1D68A, "a", 26),
        # Script
        (0x1D49C, "A", 26), (0x1D4B6, "a", 26),
        # Bold Script
        (0x1D4D0, "A", 26), (0x1D4EA, "a", 26),
        # Fraktur
        (0x1D504, "A", 26), (0x1D51E, "a", 26),
        # Bold Fraktur
        (0x1D56C, "A", 26), (0x1D586, "a", 26),
        # Double-Struck
        (0x1D538, "A", 26), (0x1D552, "a", 26),
    ]
    for start_cp, ascii_start, count in ranges:
        for i in range(count):
            m[chr(start_cp + i)] = chr(ord(ascii_start) + i)
    return m


def _build_fullwidth_map() -> dict[str, str]:
    """Fullwidth ASCII variants U+FF01–U+FF5E → standard ASCII U+0021–U+007E."""
    return {chr(cp): chr(cp - 0xFF01 + 0x21) for cp in range(0xFF01, 0xFF5F)}


def _build_homoglyph_map() -> dict[str, str]:
    m: dict[str, str] = {}
    m.update(_CYRILLIC_MAP)
    m.update(_GREEK_MAP)
    m.update(_build_math_bold_italic_map())
    m.update(_build_fullwidth_map())
    return m


_HOMOGLYPH_MAP: dict[str, str] = _build_homoglyph_map()

# Pre-compiled regex for zero-width / format characters
_EXPLICIT_ZW = set("\u200b\u200c\u200d\ufeff\u00ad\u2060\u180e")

# Bidi control characters
_BIDI_CHARS = set()
for _cp in range(0x202A, 0x202F):  # U+202A – U+202E
    _BIDI_CHARS.add(chr(_cp))
for _cp in range(0x2066, 0x206A):  # U+2066 – U+2069
    _BIDI_CHARS.add(chr(_cp))
_BIDI_CHARS.add("\u200e")  # LRM
_BIDI_CHARS.add("\u200f")  # RLM

# Emoji variation selectors
_VARIATION_SELECTORS = set()
for _cp in range(0xFE00, 0xFE10):       # U+FE00–U+FE0F
    _VARIATION_SELECTORS.add(chr(_cp))
for _cp in range(0xE0100, 0xE01F0):     # U+E0100–U+E01EF
    _VARIATION_SELECTORS.add(chr(_cp))


class Layer0Normalizer:
    """Input normalization layer. Cleans and standardizes raw user input.

    Applies in order:
      1. Zero-width / Unicode format character removal
      2. Homoglyph resolution (Cyrillic, Greek, math, fullwidth → ASCII)
      3. NFKC Unicode normalization
      4. Bidi control character stripping
      5. Emoji variation selector stripping
    """

    def __init__(self) -> None:
        self._homoglyphs = _HOMOGLYPH_MAP

    # ------------------------------------------------------------------
    # Internal normalisation steps
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_zero_width(text: str) -> tuple[str, int]:
        """Strip explicit zero-width codepoints and Unicode Cf category chars,
        but skip bidi and variation-selector chars (handled by later steps)."""
        count = 0
        out: list[str] = []
        for ch in text:
            if ch in _BIDI_CHARS or ch in _VARIATION_SELECTORS:
                out.append(ch)
            elif ch in _EXPLICIT_ZW or unicodedata.category(ch) == "Cf":
                count += 1
            else:
                out.append(ch)
        return "".join(out), count

    def _resolve_homoglyphs(self, text: str) -> tuple[str, int]:
        count = 0
        out: list[str] = []
        for ch in text:
            replacement = self._homoglyphs.get(ch)
            if replacement is not None:
                out.append(replacement)
                count += 1
            else:
                out.append(ch)
        return "".join(out), count

    @staticmethod
    def _normalize_nfkc(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _strip_bidi(text: str) -> tuple[str, int]:
        count = 0
        out: list[str] = []
        for ch in text:
            if ch in _BIDI_CHARS:
                count += 1
            else:
                out.append(ch)
        return "".join(out), count

    @staticmethod
    def _strip_variation_selectors(text: str) -> tuple[str, int]:
        count = 0
        out: list[str] = []
        for ch in text:
            if ch in _VARIATION_SELECTORS:
                count += 1
            else:
                out.append(ch)
        return "".join(out), count

    # ------------------------------------------------------------------
    # Risk signal computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_risk(
        original_len: int,
        zw_count: int,
        homoglyph_count: int,
        bidi_count: int,
        vs_count: int,
    ) -> float:
        if original_len == 0:
            return 0.0
        total_changes = zw_count + homoglyph_count + bidi_count + vs_count
        if total_changes == 0:
            return 0.0
        # ratio of changed characters to original length, scaled so heavy
        # obfuscation (>15 % of chars) reaches > 0.6
        ratio = total_changes / max(original_len, 1)
        # Apply a sigmoid-like scaling: tanh(ratio * 5) gives ~0.6 at ratio=0.14
        import math
        signal = math.tanh(ratio * 5.0)
        return round(min(signal, 1.0), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pipeline_state: dict) -> dict:
        message = pipeline_state.get("message", None)
        if message is None or not isinstance(message, str):
            pipeline_state["normalized_text"] = ""
            pipeline_state["normalization_changes"] = []
            pipeline_state["normalization_risk_signal"] = 0.0
            return pipeline_state

        if len(message) == 0:
            pipeline_state["normalized_text"] = ""
            pipeline_state["normalization_changes"] = []
            pipeline_state["normalization_risk_signal"] = 0.0
            return pipeline_state

        original_len = len(message)
        changes: list[str] = []
        text = message

        # 1. Zero-width character removal
        text, zw_count = self._remove_zero_width(text)
        if zw_count > 0:
            changes.append(f"removed {zw_count} zero-width chars")

        # 2. Homoglyph resolution
        text, hg_count = self._resolve_homoglyphs(text)
        if hg_count > 0:
            changes.append(f"resolved {hg_count} homoglyphs")

        # 3. Unicode NFKC normalization
        pre_nfkc = text
        text = self._normalize_nfkc(text)
        if text != pre_nfkc:
            changes.append("applied NFKC normalization")

        # 4. Bidi control character stripping
        text, bidi_count = self._strip_bidi(text)
        if bidi_count > 0:
            changes.append(f"stripped {bidi_count} bidi control chars")

        # 5. Emoji variation selector stripping
        text, vs_count = self._strip_variation_selectors(text)
        if vs_count > 0:
            changes.append(f"stripped {vs_count} variation selectors")

        risk = self._compute_risk(original_len, zw_count, hg_count, bidi_count, vs_count)

        pipeline_state["normalized_text"] = text
        pipeline_state["normalization_changes"] = changes
        pipeline_state["normalization_risk_signal"] = risk
        return pipeline_state
