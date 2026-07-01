"""Data models for the Blind Test provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from mashumaro import DataClassDictMixin


class BlindTestPhase(StrEnum):
    """Blind Test session phases."""

    LOBBY = "lobby"
    ANSWERING = "answering"
    REVEAL = "reveal"
    FINISHED = "finished"


@dataclass
class BlindTestConfig(DataClassDictMixin):
    """Configuration for a Blind Test session."""

    player_id: str
    round_count: int = 2
    suggestion_count: int = 4
    answer_duration: int = 30
    source_uris: list[str] = field(default_factory=list)
    name: str | None = None
    play_on_player: bool = True
    play_on_joined_players: bool = True


@dataclass
class BlindTestPlayer(DataClassDictMixin):
    """A player participating in a Blind Test session."""

    player_id: str
    name: str
    token_hash: str
    joined_at: float
    active_from_round: int
    score: int = 0
    connected: bool = True
    last_seen: float = 0
    ready: bool = False
    sendspin_player_id: str | None = None


@dataclass
class BlindTestSuggestion(DataClassDictMixin):
    """A possible answer for a Blind Test round."""

    suggestion_id: str
    label: str
    uri: str | None = None
    is_correct: bool = False


@dataclass
class BlindTestAnswer(DataClassDictMixin):
    """A locked player answer for a Blind Test round."""

    player_id: str
    suggestion_id: str
    answered_at: float
    is_correct: bool
    points: int = 0


@dataclass
class BlindTestRound(DataClassDictMixin):
    """A single Blind Test round."""

    round_index: int
    track_uri: str
    answer_label: str
    suggestions: list[BlindTestSuggestion]
    image_url: str | None = None
    duration: float | None = None
    started_at: float | None = None
    ended_at: float | None = None
    answers: dict[str, BlindTestAnswer] = field(default_factory=dict)


@dataclass
class BlindTestSession(DataClassDictMixin):
    """A Blind Test game session."""

    session_id: str
    join_code: str
    config: BlindTestConfig
    phase: BlindTestPhase = BlindTestPhase.LOBBY
    players: dict[str, BlindTestPlayer] = field(default_factory=dict)
    rounds: list[BlindTestRound] = field(default_factory=list)
    current_round_index: int | None = None
