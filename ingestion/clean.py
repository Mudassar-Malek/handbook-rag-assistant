"""Text cleaning for handbook pages.

The PDF extraction produces three recurring problems, each handled below:

1. Boilerplate lines  -> "EMPLOYEE"/"HANDBOOK" headers, bare page-number
   footers, and long underscore rules repeated on every page.
2. Run-together words -> "theCompany", "ofstakeholders", "2.1.EQUALOPPORTUNITY"
   caused by the column layout dropping spaces.
3. Inconsistent unicode / whitespace.

The de-gluing pass is intentionally *conservative* and regex-only. We do NOT
use a dictionary splitter (e.g. wordninja) because it would mangle product
names and acronyms such as AcmeApp, HRMS, and ESS.
"""

from __future__ import annotations

import re
import unicodedata

from . import config

# --- Boilerplate: lines that should be removed entirely ---------------------
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*EMPLOYEE\s*$", re.IGNORECASE),
    re.compile(r"^\s*HANDBOOK\s*$", re.IGNORECASE),
    re.compile(r"^\s*EMPLOYEE\s+HANDBOOK\s*$", re.IGNORECASE),
    # Bare page-number footer (1-3 digits on its own line).
    re.compile(r"^\s*\d{1,3}\s*$"),
    # Long underscore / dash "rule" lines (decorative separators).
    re.compile(r"^[\s_\u2014\u2015\u2013\-]{4,}$"),
]

# --- De-gluing regexes ------------------------------------------------------
# Insert a space at a lowercase -> Uppercase boundary: "theCompany" -> "the Company".
_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")
# Insert a space after . : ; when glued to a following *uppercase* letter:
# "satisfaction.We" -> "satisfaction. We", "2.1.EQUAL" -> "2.1. EQUAL".
# Restricting to an uppercase follower keeps "e.g." / "i.e." and decimals intact.
_PUNCT_GLUE = re.compile(r"([.:;])([A-Z])")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize so ligatures / full-width chars become plain ASCII-ish."""
    return unicodedata.normalize("NFKC", text)


def strip_boilerplate(text: str) -> str:
    """Drop repeated header/footer and decorative-rule lines."""
    kept = [
        line
        for line in text.splitlines()
        if not any(p.match(line) for p in _BOILERPLATE_PATTERNS)
    ]
    return "\n".join(kept)


def deglue(text: str) -> str:
    """Conservatively re-insert spaces lost to the column layout."""
    text = _LOWER_UPPER.sub(r"\1 \2", text)
    text = _PUNCT_GLUE.sub(r"\1 \2", text)
    # Restore protected product names the lowercase->Uppercase rule may have split.
    for spaced, correct in config.PROTECTED_TERMS.items():
        text = text.replace(spaced, correct)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs per line and limit blank-line runs.

    Newlines are preserved because the chunker relies on line structure to
    detect headings; only horizontal whitespace and excess blank lines go.
    """
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines down to a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page(raw: str) -> str:
    """Full per-page cleaning pipeline, order matters.

    normalize -> strip boilerplate -> de-glue -> collapse whitespace.
    """
    text = normalize_unicode(raw)
    text = strip_boilerplate(text)
    text = deglue(text)
    text = collapse_whitespace(text)
    return text


def smart_title_case(heading: str) -> str:
    """Title-case an ALL-CAPS heading while preserving known acronyms.

    "EARNED LEAVE (EL)"        -> "Earned Leave (EL)"
    "BRING YOUR OWN DEVICE"    -> "Bring Your Own Device"
    "NON-HARASSMENT / NON-DISCRIMINATION" -> "Non-Harassment / Non-Discrimination"
    """
    out_words = []
    for word in heading.split():
        # Pull out a parenthesised acronym like "(EL)" / "(SL)".
        inner = word.strip("()[].,:;")
        if inner.upper() in config.ACRONYMS:
            out_words.append(word.replace(inner, inner.upper()))
            continue
        # Hyphenated compounds: title-case each part ("non-harassment" -> "Non-Harassment").
        parts = word.split("-")
        cased = "-".join(_cap_token(p) for p in parts)
        out_words.append(cased)
    return " ".join(out_words)


def _cap_token(token: str) -> str:
    if not token:
        return token
    if token.upper() in config.ACRONYMS:
        return token.upper()
    return token[:1].upper() + token[1:].lower()
