"""Text preprocessing helpers."""

import re
import string
from collections.abc import Sequence

from .types import FeedbackRecord

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_text(text: str) -> str:
    """Normalize text with lowercasing and punctuation/whitespace cleanup."""
    lowered = text.lower().strip()
    no_punctuation = lowered.translate(_PUNCTUATION_TRANSLATION)
    return _WHITESPACE_RE.sub(" ", no_punctuation).strip()


def preprocess_records(records: Sequence[FeedbackRecord]) -> list[FeedbackRecord]:
    """Normalize text and labels from raw records."""
    return [
        FeedbackRecord(text=normalize_text(record.text), label=record.label.strip().lower())
        for record in records
    ]
