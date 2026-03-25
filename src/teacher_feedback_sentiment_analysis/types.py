"""Shared data types for the sentiment analysis package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackRecord:
    """A single labeled teacher feedback comment."""

    text: str
    label: str
