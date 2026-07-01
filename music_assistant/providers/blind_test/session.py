"""Session state helpers for the Blind Test provider."""

from __future__ import annotations

from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.blind_test.models import (
    BlindTestAnswer,
    BlindTestPhase,
    BlindTestPlayer,
    BlindTestRound,
    BlindTestSession,
)
from music_assistant.providers.blind_test.scoring import calculate_linear_scores


def add_player(
    session: BlindTestSession,
    player: BlindTestPlayer,
) -> None:
    """
    Add a player to a session.

    :param session: Session to mutate.
    :param player: Player to add.
    """
    normalized_name = player.name.casefold()
    if any(existing.name.casefold() == normalized_name for existing in session.players.values()):
        raise InvalidDataError("Player name must be unique")
    session.players[player.player_id] = player


def submit_answer(
    session: BlindTestSession,
    player_id: str,
    suggestion_id: str,
    answered_at: float,
) -> BlindTestAnswer:
    """
    Submit and lock a player answer for the current round.

    :param session: Session to mutate.
    :param player_id: Player submitting the answer.
    :param suggestion_id: Selected suggestion ID.
    :param answered_at: Answer timestamp.
    """
    if session.phase != BlindTestPhase.ANSWERING:
        raise InvalidDataError("Answers can only be submitted during the answering phase")
    current_round = get_current_round(session)
    if player_id in current_round.answers:
        raise InvalidDataError("Player already answered this round")
    if player_id not in session.players:
        raise InvalidDataError("Unknown player")
    if session.players[player_id].active_from_round > current_round.round_index:
        raise InvalidDataError("Player is not active for this round")
    suggestion = next(
        (item for item in current_round.suggestions if item.suggestion_id == suggestion_id),
        None,
    )
    if suggestion is None:
        raise InvalidDataError("Unknown suggestion")
    answer = BlindTestAnswer(
        player_id=player_id,
        suggestion_id=suggestion_id,
        answered_at=answered_at,
        is_correct=suggestion.is_correct,
    )
    current_round.answers[player_id] = answer
    return answer


def reveal_round(session: BlindTestSession) -> None:
    """
    Reveal the current round and apply scores.

    :param session: Session to mutate.
    """
    current_round = get_current_round(session)
    if session.phase != BlindTestPhase.ANSWERING:
        raise InvalidDataError("Round can only be revealed during the answering phase")
    correct_answer_order = [
        answer.player_id
        for answer in sorted(current_round.answers.values(), key=lambda item: item.answered_at)
        if answer.is_correct
    ]
    scores = calculate_linear_scores(correct_answer_order)
    for answer in current_round.answers.values():
        answer.points = scores.get(answer.player_id, 0)
        session.players[answer.player_id].score += answer.points
    current_round.ended_at = max(
        [answer.answered_at for answer in current_round.answers.values()],
        default=current_round.started_at or 0,
    )
    session.phase = BlindTestPhase.REVEAL
    for player in session.players.values():
        player.ready = False


def start_round(
    session: BlindTestSession,
    blind_test_round: BlindTestRound,
    started_at: float,
) -> None:
    """
    Start a new answering round.

    :param session: Session to mutate.
    :param blind_test_round: Round to append and make current.
    :param started_at: Round start timestamp.
    """
    if session.phase not in (BlindTestPhase.LOBBY, BlindTestPhase.REVEAL):
        raise InvalidDataError("A round cannot be started from the current phase")
    if len(session.rounds) >= session.config.round_count:
        raise InvalidDataError("All configured rounds have already been played")
    expected_index = len(session.rounds)
    if blind_test_round.round_index != expected_index:
        raise InvalidDataError("Round index does not match the session state")
    if not any(suggestion.is_correct for suggestion in blind_test_round.suggestions):
        raise InvalidDataError("Round requires a correct suggestion")
    if len(blind_test_round.suggestions) != session.config.suggestion_count:
        raise InvalidDataError("Round suggestion count does not match the session config")
    blind_test_round.started_at = started_at
    blind_test_round.ended_at = None
    session.rounds.append(blind_test_round)
    session.current_round_index = blind_test_round.round_index
    session.phase = BlindTestPhase.ANSWERING
    for player in session.players.values():
        player.ready = False


def mark_player_ready(session: BlindTestSession, player_id: str) -> None:
    """
    Mark a player ready during the reveal/listening phase.

    :param session: Session to mutate.
    :param player_id: Player ID.
    """
    if session.phase != BlindTestPhase.REVEAL:
        raise InvalidDataError("Players can only become ready during reveal")
    if player_id not in session.players:
        raise InvalidDataError("Unknown player")
    session.players[player_id].ready = True


def are_active_players_ready(session: BlindTestSession) -> bool:
    """
    Return whether every connected active player is ready.

    :param session: Session to inspect.
    """
    current_round = get_current_round(session)
    players = [
        player
        for player in active_players_for_round(session, current_round.round_index)
        if player.connected
    ]
    return bool(players) and all(player.ready for player in players)


def finish_session(session: BlindTestSession) -> None:
    """
    Mark a session finished.

    :param session: Session to mutate.
    """
    session.phase = BlindTestPhase.FINISHED


def reset_session(session: BlindTestSession) -> None:
    """
    Reset a session for a new game with the same config and players.

    :param session: Session to mutate.
    """
    session.phase = BlindTestPhase.LOBBY
    session.rounds.clear()
    session.current_round_index = None
    for player in session.players.values():
        player.score = 0
        player.ready = False
        player.active_from_round = 0


def active_players_for_round(session: BlindTestSession, round_index: int) -> list[BlindTestPlayer]:
    """
    Return players active for a round.

    :param session: Session to inspect.
    :param round_index: Round index.
    """
    return [
        player for player in session.players.values() if player.active_from_round <= round_index
    ]


def get_current_round(session: BlindTestSession) -> BlindTestRound:
    """
    Return the current round.

    :param session: Session to inspect.
    """
    if session.current_round_index is None:
        raise InvalidDataError("No current round")
    try:
        return session.rounds[session.current_round_index]
    except IndexError as err:
        raise InvalidDataError("Current round does not exist") from err
