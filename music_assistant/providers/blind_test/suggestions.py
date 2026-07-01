"""Suggestion helpers for the Blind Test provider."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from music_assistant.providers.blind_test.models import BlindTestSuggestion

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SuggestionCandidate:
    """A candidate track answer for Blind Test suggestions."""

    label: str
    uri: str | None = None


def normalize_answer_label(label: str) -> str:
    """
    Normalize an answer label for duplicate detection.

    :param label: Answer label to normalize.
    """
    return NORMALIZE_PATTERN.sub(" ", label.casefold()).strip()


def build_answer_label(artist: str | None, title: str) -> str:
    """
    Build the displayed answer label.

    :param artist: Artist name.
    :param title: Track title.
    """
    if artist:
        return f"{artist} - {title}"
    return title


def build_suggestions(
    correct: SuggestionCandidate,
    distractors: Iterable[SuggestionCandidate],
    suggestion_count: int,
    *,
    rng: random.Random | None = None,
) -> list[BlindTestSuggestion]:
    """
    Build shuffled suggestions containing exactly one correct answer.

    :param correct: Correct answer candidate.
    :param distractors: Wrong answer candidates.
    :param suggestion_count: Total number of suggestions to return.
    :param rng: Optional random generator.
    """
    if suggestion_count < 2:
        msg = "Suggestion count must be at least 2"
        raise ValueError(msg)

    selected = _select_distractors(correct, distractors, suggestion_count - 1)
    suggestions = [
        BlindTestSuggestion(
            suggestion_id="correct",
            label=correct.label,
            uri=correct.uri,
            is_correct=True,
        ),
        *[
            BlindTestSuggestion(
                suggestion_id=f"wrong_{index}",
                label=candidate.label,
                uri=candidate.uri,
            )
            for index, candidate in enumerate(selected)
        ],
    ]
    (rng or random).shuffle(suggestions)
    return suggestions


def _select_distractors(
    correct: SuggestionCandidate,
    distractors: Iterable[SuggestionCandidate],
    needed_count: int,
) -> Sequence[SuggestionCandidate]:
    """Return unique distractors that do not match the correct answer."""
    correct_label = normalize_answer_label(correct.label)
    seen_labels = {correct_label}
    seen_uris = {correct.uri} if correct.uri else set()
    selected: list[SuggestionCandidate] = []
    for candidate in distractors:
        candidate_label = normalize_answer_label(candidate.label)
        if not candidate_label or candidate_label in seen_labels:
            continue
        if candidate.uri and candidate.uri in seen_uris:
            continue
        seen_labels.add(candidate_label)
        if candidate.uri:
            seen_uris.add(candidate.uri)
        selected.append(candidate)
        if len(selected) == needed_count:
            break
    if len(selected) < needed_count:
        msg = "Not enough distractors to build suggestions"
        raise ValueError(msg)
    return selected
