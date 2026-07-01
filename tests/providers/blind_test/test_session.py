"""Tests for Blind Test session helpers."""

from __future__ import annotations

import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.blind_test.models import (
    BlindTestConfig,
    BlindTestPhase,
    BlindTestPlayer,
    BlindTestRound,
    BlindTestSession,
    BlindTestSuggestion,
)
from music_assistant.providers.blind_test.session import (
    active_players_for_round,
    add_player,
    reset_session,
    reveal_round,
    submit_answer,
)


def _player(player_id: str, name: str, active_from_round: int = 0) -> BlindTestPlayer:
    """Return a test player."""
    return BlindTestPlayer(
        player_id=player_id,
        name=name,
        token_hash=f"token_{player_id}",
        joined_at=1,
        last_seen=1,
        active_from_round=active_from_round,
    )


def _session() -> BlindTestSession:
    """Return a test session with one active round."""
    return BlindTestSession(
        session_id="session",
        join_code="join",
        config=BlindTestConfig(player_id="queue_1"),
        phase=BlindTestPhase.ANSWERING,
        current_round_index=0,
        rounds=[
            BlindTestRound(
                round_index=0,
                track_uri="library://track/1",
                answer_label="Daft Punk - One More Time",
                suggestions=[
                    BlindTestSuggestion(
                        suggestion_id="correct",
                        label="Daft Punk - One More Time",
                        is_correct=True,
                    ),
                    BlindTestSuggestion(
                        suggestion_id="wrong_1",
                        label="Justice - D.A.N.C.E.",
                    ),
                ],
            )
        ],
    )


def test_add_player_rejects_duplicate_name_case_insensitive() -> None:
    """Reject duplicate player names case-insensitively."""
    session = _session()
    add_player(session, _player("p1", "Alice"))

    with pytest.raises(InvalidDataError, match="unique"):
        add_player(session, _player("p2", "alice"))


def test_submit_answer_locks_first_answer() -> None:
    """Reject a second answer from the same player."""
    session = _session()
    add_player(session, _player("p1", "Alice"))

    answer = submit_answer(session, "p1", "wrong_1", 10)

    assert answer.suggestion_id == "wrong_1"
    assert answer.is_correct is False
    with pytest.raises(InvalidDataError, match="already answered"):
        submit_answer(session, "p1", "correct", 11)


def test_submit_answer_rejects_unknown_suggestion() -> None:
    """Reject answers for suggestions that are not in the current round."""
    session = _session()
    add_player(session, _player("p1", "Alice"))

    with pytest.raises(InvalidDataError, match="Unknown suggestion"):
        submit_answer(session, "p1", "missing", 10)


def test_submit_answer_rejects_outside_answering_phase() -> None:
    """Only accept answers during the answering phase."""
    session = _session()
    session.phase = BlindTestPhase.REVEAL
    add_player(session, _player("p1", "Alice"))

    with pytest.raises(InvalidDataError, match="answering phase"):
        submit_answer(session, "p1", "correct", 10)


def test_reveal_round_applies_scores_to_correct_answers_in_order() -> None:
    """Apply linear scores when a round is revealed."""
    session = _session()
    add_player(session, _player("p1", "Alice"))
    add_player(session, _player("p2", "Bob"))
    add_player(session, _player("p3", "Charlie"))
    submit_answer(session, "p2", "correct", 10)
    submit_answer(session, "p1", "wrong_1", 11)
    submit_answer(session, "p3", "correct", 12)

    reveal_round(session)

    current_round = session.rounds[0]
    assert session.phase == BlindTestPhase.REVEAL
    assert current_round.answers["p2"].points == 1000
    assert current_round.answers["p3"].points == 500
    assert current_round.answers["p1"].points == 0
    assert session.players["p2"].score == 1000
    assert session.players["p3"].score == 500
    assert session.players["p1"].score == 0


def test_active_players_for_round_excludes_late_joiners_until_next_round() -> None:
    """Only include late joiners from their active round onward."""
    session = _session()
    add_player(session, _player("p1", "Alice", active_from_round=0))
    add_player(session, _player("p2", "Bob", active_from_round=1))

    assert [player.player_id for player in active_players_for_round(session, 0)] == ["p1"]
    assert [player.player_id for player in active_players_for_round(session, 1)] == [
        "p1",
        "p2",
    ]


def test_reset_session_keeps_players_and_config_for_new_game() -> None:
    """Reset rounds, scores, readiness and late-join state for a fresh game."""
    session = _session()
    add_player(session, _player("p1", "Alice", active_from_round=0))
    add_player(session, _player("p2", "Bob", active_from_round=1))
    session.players["p1"].score = 1000
    session.players["p1"].ready = True
    session.players["p2"].score = 500
    session.players["p2"].ready = True

    reset_session(session)

    assert session.phase == BlindTestPhase.LOBBY
    assert session.current_round_index is None
    assert session.rounds == []
    assert session.config.player_id == "queue_1"
    assert {player.name for player in session.players.values()} == {"Alice", "Bob"}
    assert all(player.score == 0 for player in session.players.values())
    assert all(player.ready is False for player in session.players.values())
    assert all(player.active_from_round == 0 for player in session.players.values())
