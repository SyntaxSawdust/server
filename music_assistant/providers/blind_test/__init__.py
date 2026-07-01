"""
Blind Test Plugin Provider for Music Assistant.

Provides the backend game engine for multiplayer blind test sessions.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Playlist, Track

from music_assistant.models.plugin import PluginProvider
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
    are_active_players_ready,
    finish_session,
    get_current_round,
    mark_player_ready,
    reset_session,
    reveal_round,
    start_round,
    submit_answer,
)
from music_assistant.providers.blind_test.suggestions import (
    SuggestionCandidate,
    build_answer_label,
    build_suggestions,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return BlindTestPlugin(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """
    Return Config entries to setup this provider.

    Blind Test V1 is configured per game from the frontend, so the provider has
    no persistent settings yet.
    """
    return ()


class BlindTestPlugin(PluginProvider):
    """Blind Test plugin provider for Music Assistant."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature],
    ) -> None:
        """Initialize the Blind Test plugin."""
        super().__init__(mass, manifest, config, supported_features)
        self._sessions: dict[str, BlindTestSession] = {}
        self._unregister_handles: list[Callable[[], None]] = []

    async def loaded_in_mass(self) -> None:
        """Call after the provider has been loaded."""
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/create",
                self.create_session,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/session",
                self.get_session,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/join",
                self.join_session,
                authenticated=False,
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/state",
                self.get_player_state,
                authenticated=False,
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/start",
                self.start_session,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/prepare_round",
                self.prepare_round,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/answer",
                self.answer,
                authenticated=False,
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/reveal",
                self.reveal,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/ready",
                self.ready,
                authenticated=False,
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/next",
                self.next_round,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/reset",
                self.reset_session,
                required_role="user",
            )
        )

    async def create_session(
        self,
        player_id: str,
        round_count: int = 2,
        suggestion_count: int = 4,
        answer_duration: int = 30,
        source_uris: list[str] | None = None,
        name: str | None = None,
        play_on_player: bool = True,
        play_on_joined_players: bool = True,
    ) -> dict[str, Any]:
        """
        Create a Blind Test session.

        :param player_id: Music Assistant player/queue ID to use for playback.
        :param round_count: Number of rounds to play.
        :param suggestion_count: Number of answer suggestions per round.
        :param answer_duration: Answering duration in seconds.
        :param source_uris: Track or playlist URIs selected by the host.
        :param name: Optional game name.
        :param play_on_player: Play rounds on the configured Music Assistant player.
        :param play_on_joined_players: Play rounds on joined phone Sendspin players.
        """
        config = BlindTestConfig(
            player_id=player_id,
            round_count=round_count,
            suggestion_count=suggestion_count,
            answer_duration=answer_duration,
            source_uris=source_uris or [],
            name=name,
            play_on_player=play_on_player,
            play_on_joined_players=play_on_joined_players,
        )
        _validate_config(config)
        session = BlindTestSession(
            session_id=secrets.token_urlsafe(12),
            join_code=secrets.token_urlsafe(8),
            config=config,
        )
        self._sessions[session.session_id] = session
        return _host_state(session)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Return host-visible session state.

        :param session_id: Session ID to fetch.
        """
        session = self._get_session(session_id)
        self._sync_answering_phase(session)
        return _host_state(session)

    async def join_session(
        self,
        session_id: str,
        name: str,
        sendspin_player_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Join a Blind Test session.

        :param session_id: Session ID to join.
        :param name: Unique player display name.
        :param sendspin_player_id: Temporary Sendspin player ID for this phone.
        """
        session = self._get_session(session_id)
        player_name = name.strip()
        if not player_name:
            raise InvalidDataError("Player name is required")
        player_token = secrets.token_urlsafe(24)
        player = BlindTestPlayer(
            player_id=secrets.token_urlsafe(12),
            name=player_name,
            token_hash=_hash_token(player_token),
            joined_at=time.time(),
            last_seen=time.time(),
            active_from_round=_get_join_round(session),
            sendspin_player_id=_clean_sendspin_player_id(sendspin_player_id),
        )
        add_player(session, player)
        await self._attach_player_to_current_playback(session, player)
        return {
            "player_id": player.player_id,
            "player_token": player_token,
            "state": _player_state(session, player),
        }

    async def get_player_state(
        self,
        session_id: str,
        player_token: str,
        sendspin_player_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Return player-visible session state.

        :param session_id: Session ID to fetch.
        :param player_token: Opaque player reconnect token.
        :param sendspin_player_id: Temporary Sendspin player ID for this phone.
        """
        session = self._get_session(session_id)
        self._sync_answering_phase(session)
        player = _get_player_by_token(session, player_token)
        player.last_seen = time.time()
        player.connected = True
        if sendspin_player_id:
            player.sendspin_player_id = _clean_sendspin_player_id(sendspin_player_id)
            await self._attach_player_to_current_playback(session, player)
        return _player_state(session, player)

    async def start_session(
        self,
        session_id: str,
        round_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start the first Blind Test round.

        :param session_id: Session ID to start.
        :param round_payload: Prepared round payload.
        """
        session = self._get_session(session_id)
        blind_test_round = _round_from_payload(
            session,
            round_payload or await self.prepare_round(session_id),
        )
        start_round(session, blind_test_round, time.time())
        await self._play_round(session)
        return _host_state(session)

    async def prepare_round(self, session_id: str) -> dict[str, Any]:
        """
        Prepare the next Blind Test round from the configured sources.

        :param session_id: Session ID.
        """
        session = self._get_session(session_id)
        track = await self._get_next_source_track(session)
        correct = _track_to_candidate(track)
        search_results = await self.mass.music.search(
            search_query=correct.label,
            media_types=[MediaType.TRACK],
            limit=max(session.config.suggestion_count * 3, 12),
            library_only=False,
        )
        distractors = [
            _track_to_candidate(item) for item in search_results.tracks if isinstance(item, Track)
        ]
        try:
            suggestions = build_suggestions(
                correct,
                distractors,
                session.config.suggestion_count,
            )
        except ValueError as err:
            raise InvalidDataError(str(err)) from err
        image_url = await self.mass.metadata.get_image_url_for_item(track)
        return {
            "track_uri": track.uri,
            "answer_label": correct.label,
            "image_url": image_url,
            "duration": track.duration,
            "suggestions": [suggestion.to_dict() for suggestion in suggestions],
        }

    async def answer(
        self,
        session_id: str,
        player_token: str,
        suggestion_id: str,
    ) -> dict[str, Any]:
        """
        Submit and lock a player's answer.

        :param session_id: Session ID.
        :param player_token: Opaque player token.
        :param suggestion_id: Selected suggestion ID.
        """
        session = self._get_session(session_id)
        self._sync_answering_phase(session)
        player = _get_player_by_token(session, player_token)
        submit_answer(session, player.player_id, suggestion_id, time.time())
        player.last_seen = time.time()
        self._sync_answering_phase(session)
        return _player_state(session, player)

    async def reveal(self, session_id: str) -> dict[str, Any]:
        """
        Reveal the current round and apply scoring.

        :param session_id: Session ID.
        """
        session = self._get_session(session_id)
        reveal_round(session)
        return _host_state(session)

    async def ready(self, session_id: str, player_token: str) -> dict[str, Any]:
        """
        Mark a player ready for the next round.

        :param session_id: Session ID.
        :param player_token: Opaque player token.
        """
        session = self._get_session(session_id)
        player = _get_player_by_token(session, player_token)
        mark_player_ready(session, player.player_id)
        player.last_seen = time.time()
        if are_active_players_ready(session):
            await self._advance_from_reveal(session)
        return _player_state(session, player)

    async def next_round(
        self,
        session_id: str,
        round_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Advance to the next round or finish the game.

        :param session_id: Session ID.
        :param round_payload: Prepared payload for the next round.
        """
        session = self._get_session(session_id)
        if session.phase != BlindTestPhase.REVEAL:
            raise InvalidDataError("Next round can only start after reveal")
        await self._advance_from_reveal(session, round_payload)
        return _host_state(session)

    async def reset_session(self, session_id: str) -> dict[str, Any]:
        """
        Reset a Blind Test session for a new game with the same settings.

        :param session_id: Session ID.
        """
        session = self._get_session(session_id)
        await self._stop_playback(session)
        reset_session(session)
        return _host_state(session)

    async def _advance_from_reveal(
        self,
        session: BlindTestSession,
        round_payload: dict[str, Any] | None = None,
    ) -> None:
        """Advance a revealed session to the next round or finish it."""
        if len(session.rounds) >= session.config.round_count:
            await self._stop_playback(session)
            finish_session(session)
            return
        if round_payload is None:
            round_payload = await self.prepare_round(session.session_id)
        blind_test_round = _round_from_payload(session, round_payload)
        start_round(session, blind_test_round, time.time())
        await self._play_round(session)

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        :param is_removed: Whether the provider is being removed.
        """
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        self._sessions.clear()
        await super().unload(is_removed)

    def _get_session(self, session_id: str) -> BlindTestSession:
        """Return a session by ID."""
        try:
            return self._sessions[session_id]
        except KeyError as err:
            raise InvalidDataError("Unknown Blind Test session") from err

    def _sync_answering_phase(self, session: BlindTestSession) -> None:
        """Reveal an answering round when answers are complete or time is up."""
        if session.phase != BlindTestPhase.ANSWERING:
            return
        current_round = get_current_round(session)
        if current_round.started_at is None:
            return
        now = time.time()
        active_connected_players = [
            player
            for player in active_players_for_round(session, current_round.round_index)
            if player.connected
        ]
        all_active_players_answered = bool(active_connected_players) and all(
            player.player_id in current_round.answers for player in active_connected_players
        )
        round_duration = float(session.config.answer_duration)
        if current_round.duration and current_round.duration > 0:
            round_duration = min(round_duration, current_round.duration)
        answer_deadline_reached = now >= current_round.started_at + round_duration
        if all_active_players_answered or answer_deadline_reached:
            reveal_round(session)

    async def _play_round(self, session: BlindTestSession) -> None:
        """Start playback for the current round."""
        current_round = session.rounds[session.current_round_index or 0]
        playback_targets: list[tuple[str, bool]] = []
        selected_player_id = self._get_selected_playback_player_id(session)
        phone_player_ids: set[str] = set()
        if session.config.play_on_joined_players:
            phone_player_ids = {
                player.sendspin_player_id
                for player in session.players.values()
                if player.connected and player.sendspin_player_id
            }
            if session.config.play_on_player:
                phone_player_ids.discard(session.config.player_id)

        if selected_player_id:
            grouped_phone_ids = await self._sync_playback_targets(
                selected_player_id,
                phone_player_ids,
            )
            phone_player_ids -= grouped_phone_ids
            playback_targets.append((selected_player_id, False))
        elif phone_player_ids:
            selected_player_id = sorted(phone_player_ids)[0]
            phone_player_ids.remove(selected_player_id)
            grouped_phone_ids = await self._sync_playback_targets(
                selected_player_id,
                phone_player_ids,
            )
            phone_player_ids -= grouped_phone_ids
            playback_targets.append((selected_player_id, True))

        playback_targets.extend((player_id, True) for player_id in sorted(phone_player_ids))
        if not playback_targets:
            raise InvalidDataError("No playback targets are available")

        playback_errors: list[Exception] = []
        for player_id, is_phone_player in playback_targets:
            try:
                await self.mass.player_queues.play_media(
                    queue_id=player_id,
                    media=current_round.track_uri,
                )
            except Exception as err:
                playback_errors.append(err)
                player_type = "temporary phone" if is_phone_player else "selected"
                self.logger.warning(
                    "Failed to play Blind Test round on %s player %s: %s",
                    player_type,
                    player_id,
                    err,
                )
        if len(playback_errors) == len(playback_targets):
            raise playback_errors[0]

    async def _stop_playback(self, session: BlindTestSession) -> None:
        """Stop every playback target used by the session."""
        player_ids = self._get_playback_target_ids(session)
        for player_id in sorted(player_ids):
            try:
                await self.mass.player_queues.stop(player_id)
            except Exception as err:
                self.logger.warning(
                    "Failed to stop Blind Test playback on player %s: %s",
                    player_id,
                    err,
                )

    async def _sync_playback_targets(self, leader_id: str, child_ids: set[str]) -> set[str]:
        """Group compatible playback targets so Sendspin can keep them in sync."""
        groupable_child_ids = [
            child_id
            for child_id in sorted(child_ids)
            if self._can_group_players(leader_id, child_id)
        ]
        if not groupable_child_ids:
            return set()
        try:
            await self.mass.players.cmd_set_members(
                target_player=leader_id,
                player_ids_to_add=groupable_child_ids,
            )
        except Exception as err:
            self.logger.warning(
                "Failed to sync Blind Test playback targets %s to %s: %s",
                groupable_child_ids,
                leader_id,
                err,
            )
            return set()
        return set(groupable_child_ids)

    def _get_playback_target_ids(self, session: BlindTestSession) -> set[str]:
        """Return all configured and temporary playback target IDs for a session."""
        player_ids: set[str] = set()
        if selected_player_id := self._get_selected_playback_player_id(session):
            player_ids.add(selected_player_id)
        if session.config.play_on_joined_players:
            player_ids.update(
                player.sendspin_player_id
                for player in session.players.values()
                if player.sendspin_player_id
            )
        return player_ids

    async def _attach_player_to_current_playback(
        self,
        session: BlindTestSession,
        player: BlindTestPlayer,
    ) -> None:
        """Attach a late joined phone player to the current synced playback group."""
        if (
            not session.config.play_on_joined_players
            or not player.connected
            or not player.sendspin_player_id
            or session.phase not in (BlindTestPhase.ANSWERING, BlindTestPhase.REVEAL)
        ):
            return
        leader_id = self._get_current_playback_leader_id(session, player.sendspin_player_id)
        if not leader_id or not self._can_group_players(leader_id, player.sendspin_player_id):
            return
        await self._sync_playback_targets(leader_id, {player.sendspin_player_id})

    async def _get_next_source_track(self, session: BlindTestSession) -> Track:
        """Return a random unused track from configured sources."""
        if not session.config.source_uris:
            raise InvalidDataError("At least one source URI is required")
        used_track_uris = {blind_test_round.track_uri for blind_test_round in session.rounds}
        candidates: dict[str, Track] = {}
        for source_uri in session.config.source_uris:
            media_item = await self.mass.music.get_item_by_uri(source_uri)
            if isinstance(media_item, Track):
                if media_item.uri and media_item.uri not in used_track_uris:
                    candidates[media_item.uri] = media_item
                continue
            if isinstance(media_item, Playlist):
                async for track in self.mass.music.playlists.tracks(
                    item_id=media_item.item_id,
                    provider_instance_id_or_domain=media_item.provider,
                ):
                    if isinstance(track, Track) and track.uri and track.uri not in used_track_uris:
                        candidates[track.uri] = track
        if candidates:
            return secrets.choice(list(candidates.values()))
        raise InvalidDataError("No unused source tracks are available")

    def _can_group_players(self, leader_id: str, child_id: str) -> bool:
        """Return whether MA reports that the child can join the leader."""
        leader = self.mass.players.get_player(leader_id)
        child = self.mass.players.get_player(child_id)
        can_group_with = getattr(leader, "can_group_with", None)
        if can_group_with is None and (state := getattr(leader, "state", None)):
            can_group_with = getattr(state, "can_group_with", None)
        if not isinstance(can_group_with, list | tuple | set):
            return False
        child_provider_id = getattr(getattr(child, "provider", None), "instance_id", None)
        return child_id in can_group_with or child_provider_id in can_group_with

    def _get_selected_playback_player_id(self, session: BlindTestSession) -> str | None:
        """Return the selected host playback player if it can be used as a stable output."""
        if not session.config.play_on_player or not session.config.player_id:
            return None
        if _is_blind_test_sendspin_player_id(session.config.player_id):
            self.logger.warning(
                "Ignoring stale temporary Blind Test player %s as selected playback output",
                session.config.player_id,
            )
            return None
        return session.config.player_id

    def _get_current_playback_leader_id(
        self,
        session: BlindTestSession,
        child_player_id: str,
    ) -> str | None:
        """Return the current playback leader for a late joined phone player."""
        selected_player_id = self._get_selected_playback_player_id(session)
        if selected_player_id and selected_player_id != child_player_id:
            return selected_player_id
        phone_player_ids = sorted(
            player.sendspin_player_id
            for player in session.players.values()
            if player.connected
            and player.sendspin_player_id
            and player.sendspin_player_id != child_player_id
        )
        return phone_player_ids[0] if phone_player_ids else None


def _validate_config(config: BlindTestConfig) -> None:
    """Validate session config."""
    if not config.play_on_player and not config.play_on_joined_players:
        raise InvalidDataError("At least one playback output is required")
    if config.play_on_player and not config.player_id:
        raise InvalidDataError("Player is required")
    if config.round_count < 2:
        raise InvalidDataError("Blind Test requires at least 2 rounds")
    if config.suggestion_count < 2:
        raise InvalidDataError("Suggestion count must be at least 2")
    if config.answer_duration < 1:
        raise InvalidDataError("Answer duration must be at least 1 second")


def _get_join_round(session: BlindTestSession) -> int:
    """Return the first round a newly joined player may answer."""
    if session.phase == BlindTestPhase.LOBBY:
        return 0
    if session.current_round_index is None:
        return len(session.rounds)
    return session.current_round_index + 1


def _clean_sendspin_player_id(sendspin_player_id: str | None) -> str | None:
    """Return a safe temporary Sendspin player ID."""
    if not sendspin_player_id:
        return None
    cleaned = "".join(
        char for char in sendspin_player_id.strip() if char.isalnum() or char in {"_", "-"}
    )
    return cleaned[:96] or None


def _is_blind_test_sendspin_player_id(player_id: str) -> bool:
    """Return whether a player ID belongs to a temporary Blind Test Sendspin client."""
    return player_id.startswith("blind_test_")


def _round_from_payload(
    session: BlindTestSession,
    payload: dict[str, Any],
) -> BlindTestRound:
    """Create a round model from an API payload."""
    suggestions = [
        BlindTestSuggestion(
            suggestion_id=str(item["suggestion_id"]),
            label=str(item["label"]),
            uri=item.get("uri"),
            is_correct=bool(item.get("is_correct", False)),
        )
        for item in payload.get("suggestions", [])
    ]
    if len({suggestion.suggestion_id for suggestion in suggestions}) != len(suggestions):
        raise InvalidDataError("Suggestion IDs must be unique")
    return BlindTestRound(
        round_index=len(session.rounds),
        track_uri=str(payload["track_uri"]),
        answer_label=str(payload["answer_label"]),
        image_url=payload.get("image_url"),
        duration=payload.get("duration"),
        suggestions=suggestions,
    )


def _host_state(session: BlindTestSession) -> dict[str, Any]:
    """Return host-visible session state."""
    return session.to_dict()


def _player_state(session: BlindTestSession, player: BlindTestPlayer) -> dict[str, Any]:
    """Return player-visible session state."""
    state = session.to_dict()
    state["players"] = {
        player_id: {
            "player_id": item["player_id"],
            "name": item["name"],
            "score": item["score"],
            "connected": item["connected"],
            "ready": item["ready"],
            "active_from_round": item["active_from_round"],
            "sendspin_player_id": item.get("sendspin_player_id"),
        }
        for player_id, item in state["players"].items()
    }
    state["current_player_id"] = player.player_id
    if session.phase == BlindTestPhase.ANSWERING:
        for round_state in state["rounds"]:
            for suggestion in round_state["suggestions"]:
                suggestion.pop("is_correct", None)
    return state


def _get_player_by_token(session: BlindTestSession, player_token: str) -> BlindTestPlayer:
    """Return a player by token."""
    token_hash = _hash_token(player_token)
    for player in session.players.values():
        if secrets.compare_digest(player.token_hash, token_hash):
            return player
    raise InvalidDataError("Unknown Blind Test player")


def _hash_token(token: str) -> str:
    """Hash an opaque player token."""
    return hashlib.sha256(token.encode()).hexdigest()


def _track_to_candidate(track: Track) -> SuggestionCandidate:
    """Convert a track to an answer suggestion candidate."""
    return SuggestionCandidate(
        label=build_answer_label(track.artist_str or None, track.name),
        uri=track.uri,
    )
