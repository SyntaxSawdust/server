"""Tests for Blind Test suggestion helpers."""

from __future__ import annotations

import random

import pytest

from music_assistant.providers.blind_test.suggestions import (
    SuggestionCandidate,
    build_answer_label,
    build_suggestions,
    normalize_answer_label,
)


def test_build_answer_label_with_artist_and_title() -> None:
    """Build the v1 Artist - Track title answer label."""
    assert build_answer_label("Massive Attack", "Teardrop") == "Massive Attack - Teardrop"


def test_build_answer_label_without_artist() -> None:
    """Fall back to title when the artist is unknown."""
    assert build_answer_label(None, "Untitled") == "Untitled"


def test_normalize_answer_label_ignores_case_and_punctuation() -> None:
    """Normalize answer labels for duplicate detection."""
    assert normalize_answer_label("Daft Punk - One More Time!") == "daft punk one more time"


def test_build_suggestions_includes_one_correct_answer() -> None:
    """Build suggestions with exactly one correct answer."""
    suggestions = build_suggestions(
        SuggestionCandidate("Daft Punk - One More Time", "library://track/1"),
        [
            SuggestionCandidate("Justice - D.A.N.C.E.", "library://track/2"),
            SuggestionCandidate("Phoenix - Lisztomania", "library://track/3"),
            SuggestionCandidate("Air - Sexy Boy", "library://track/4"),
        ],
        4,
        rng=random.Random(1),
    )

    assert len(suggestions) == 4
    assert sum(item.is_correct for item in suggestions) == 1
    assert {item.label for item in suggestions} == {
        "Daft Punk - One More Time",
        "Justice - D.A.N.C.E.",
        "Phoenix - Lisztomania",
        "Air - Sexy Boy",
    }


def test_build_suggestions_filters_duplicate_uri_and_label() -> None:
    """Skip distractors that duplicate the correct answer or each other."""
    suggestions = build_suggestions(
        SuggestionCandidate("Daft Punk - One More Time", "library://track/1"),
        [
            SuggestionCandidate("Daft Punk - One More Time", "library://track/other"),
            SuggestionCandidate("Different label", "library://track/1"),
            SuggestionCandidate("Justice - D.A.N.C.E.", "library://track/2"),
            SuggestionCandidate("Justice D A N C E", "library://track/3"),
            SuggestionCandidate("Phoenix - Lisztomania", "library://track/4"),
        ],
        3,
        rng=random.Random(1),
    )

    assert {item.label for item in suggestions} == {
        "Daft Punk - One More Time",
        "Justice - D.A.N.C.E.",
        "Phoenix - Lisztomania",
    }


def test_build_suggestions_requires_enough_distractors() -> None:
    """Fail clearly when there are not enough unique distractors."""
    with pytest.raises(ValueError, match="Not enough distractors"):
        build_suggestions(
            SuggestionCandidate("Daft Punk - One More Time", "library://track/1"),
            [SuggestionCandidate("Daft Punk - One More Time", "library://track/2")],
            3,
        )


def test_build_suggestions_requires_at_least_two_choices() -> None:
    """Reject a suggestion count below two."""
    with pytest.raises(ValueError, match="at least 2"):
        build_suggestions(
            SuggestionCandidate("Daft Punk - One More Time", "library://track/1"),
            [],
            1,
        )
