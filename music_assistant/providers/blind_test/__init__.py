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

from music_assistant_models.enums import EventType, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import Playlist, Track

from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.blind_test.models import (
    BlindTestConfig,
    BlindTestPhase,
    BlindTestPlayer,
    BlindTestRound,
    BlindTestSession,
    BlindTestSource,
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
EVENT_SESSION_REMOVED = "blind_test_session_removed"


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
        self._lyrics_tasks: set[tuple[str, int, str]] = set()
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
                "blind_test/sessions",
                self.list_sessions,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/rename",
                self.rename_session,
                required_role="user",
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/info",
                self.get_session_info,
                authenticated=False,
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
        self._unregister_handles.append(
            self.mass.register_api_command(
                "blind_test/delete",
                self.delete_session,
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
            name=_clean_session_name(name),
            play_on_player=play_on_player,
            play_on_joined_players=play_on_joined_players,
        )
        _validate_config(config)
        session = BlindTestSession(
            session_id=secrets.token_urlsafe(12),
            join_code=secrets.token_urlsafe(8),
            config=config,
            sources=await self._resolve_session_sources(config.source_uris),
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._sessions[session.session_id] = session
        return _host_state(session)

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return live Blind Test session summaries."""
        summaries: list[dict[str, Any]] = []
        for session in self._sessions.values():
            self._sync_answering_phase(session)
            summaries.append(_session_summary(session))
        return sorted(
            summaries,
            key=lambda item: float(item["updated_at"]),
            reverse=True,
        )

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """
        Return host-visible session state.

        :param session_id: Session ID to fetch.
        """
        session = self._get_session(session_id)
        self._sync_answering_phase_and_schedule_lyrics(session)
        return _host_state(session)

    async def rename_session(self, session_id: str, name: str | None = None) -> dict[str, Any]:
        """
        Rename a Blind Test session.

        :param session_id: Session ID to rename.
        :param name: New display name.
        """
        session = self._get_session(session_id)
        session.config.name = _clean_session_name(name)
        _touch_session(session)
        return _host_state(session)

    async def get_session_info(self, session_id: str) -> dict[str, Any]:
        """
        Return public metadata for a Blind Test session.

        :param session_id: Session ID to fetch.
        """
        session = self._get_session(session_id)
        self._sync_answering_phase_and_schedule_lyrics(session)
        return _session_info(session)

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
        _touch_session(session)
        await self._attach_player_to_current_playback(session, player)
        self._schedule_current_round_lyrics_hydration(session)
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
        self._sync_answering_phase_and_schedule_lyrics(session)
        player = _get_player_by_token(session, player_token)
        player.last_seen = time.time()
        player.connected = True
        _touch_session(session)
        if sendspin_player_id:
            cleaned_sendspin_player_id = _clean_sendspin_player_id(sendspin_player_id)
            if cleaned_sendspin_player_id != player.sendspin_player_id:
                player.sendspin_player_id = cleaned_sendspin_player_id
                await self._attach_player_to_current_playback(session, player)
        self._schedule_current_round_lyrics_hydration(session)
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
        _touch_session(session)
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
            limit=max(session.config.suggestion_count * 8, 24),
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
        self._sync_answering_phase_and_schedule_lyrics(session)
        player = _get_player_by_token(session, player_token)
        submit_answer(session, player.player_id, suggestion_id, time.time())
        player.last_seen = time.time()
        _touch_session(session)
        self._sync_answering_phase_and_schedule_lyrics(session)
        return _player_state(session, player)

    async def reveal(self, session_id: str) -> dict[str, Any]:
        """
        Reveal the current round and apply scoring.

        :param session_id: Session ID.
        """
        session = self._get_session(session_id)
        reveal_round(session)
        self._schedule_current_round_lyrics_hydration(session)
        _touch_session(session)
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
        _touch_session(session)
        if are_active_players_ready(session):
            await self._advance_from_reveal(session)
        self._schedule_current_round_lyrics_hydration(session)
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
        _touch_session(session)
        return _host_state(session)

    async def delete_session(self, session_id: str) -> dict[str, str]:
        """
        Delete a Blind Test session and stop its playback targets.

        :param session_id: Session ID.
        """
        session = self._get_session(session_id)
        await self._stop_playback(session)
        self._sessions.pop(session_id, None)
        self.mass.signal_event(
            EventType.UNKNOWN,
            object_id=session_id,
            data={"type": EVENT_SESSION_REMOVED, "session_id": session_id},
        )
        return {"session_id": session_id}

    async def _advance_from_reveal(
        self,
        session: BlindTestSession,
        round_payload: dict[str, Any] | None = None,
    ) -> None:
        """Advance a revealed session to the next round or finish it."""
        if len(session.rounds) >= session.config.round_count:
            await self._stop_playback(session)
            finish_session(session)
            _touch_session(session)
            return
        if round_payload is None:
            round_payload = await self.prepare_round(session.session_id)
        blind_test_round = _round_from_payload(session, round_payload)
        start_round(session, blind_test_round, time.time())
        _touch_session(session)
        await self._play_round(session)

    async def unload(self, is_removed: bool = False) -> None:
        """
        Handle unload/close of the provider.

        :param is_removed: Whether the provider is being removed.
        """
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        self._lyrics_tasks.clear()
        self._sessions.clear()
        await super().unload(is_removed)

    def _get_session(self, session_id: str) -> BlindTestSession:
        """Return a session by ID."""
        try:
            return self._sessions[session_id]
        except KeyError as err:
            raise InvalidDataError("Unknown Blind Test session") from err

    async def _resolve_session_sources(self, source_uris: list[str]) -> list[BlindTestSource]:
        """Resolve configured source URIs into host-visible source metadata."""
        sources: list[BlindTestSource] = []
        for source_uri in source_uris:
            try:
                media_item = await self.mass.music.get_item_by_uri(source_uri)
            except Exception as err:
                self.logger.debug("Could not resolve Blind Test source %s: %s", source_uri, err)
                sources.append(BlindTestSource(uri=source_uri, name=source_uri))
                continue

            media_type = getattr(media_item, "media_type", None)
            sources.append(
                BlindTestSource(
                    uri=source_uri,
                    name=getattr(media_item, "name", source_uri) or source_uri,
                    media_type=media_type.value
                    if isinstance(media_type, MediaType)
                    else media_type,
                )
            )
        return sources

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
            _touch_session(session)

    def _sync_answering_phase_and_schedule_lyrics(self, session: BlindTestSession) -> None:
        """Sync automatic reveal state and schedule reveal-only lyrics."""
        self._sync_answering_phase(session)
        self._schedule_current_round_lyrics_hydration(session)

    def _schedule_current_round_lyrics_hydration(self, session: BlindTestSession) -> None:
        """Fetch current reveal lyrics in the background without blocking players."""
        if session.phase != BlindTestPhase.REVEAL or session.current_round_index is None:
            return
        current_round = get_current_round(session)
        if current_round.lyrics_loaded:
            return
        task_key = (session.session_id, current_round.round_index, current_round.track_uri)
        if task_key in self._lyrics_tasks:
            return
        self._lyrics_tasks.add(task_key)
        track_hash = hashlib.sha1(current_round.track_uri.encode()).hexdigest()[:10]
        try:
            self.mass.create_task(
                self._hydrate_current_round_lyrics,
                session.session_id,
                current_round.round_index,
                current_round.track_uri,
                task_id=f"blind_test_lyrics_{session.session_id}_{current_round.round_index}_{track_hash}",
            )
        except Exception:
            self._lyrics_tasks.discard(task_key)
            raise

    async def _hydrate_current_round_lyrics(
        self,
        session_id: str,
        round_index: int,
        track_uri: str,
    ) -> None:
        """Fetch lyrics for a revealed round once."""
        task_key = (session_id, round_index, track_uri)
        try:
            session = self._sessions.get(session_id)
            if (
                session is None
                or session.phase != BlindTestPhase.REVEAL
                or session.current_round_index != round_index
            ):
                return
            current_round = get_current_round(session)
            if current_round.track_uri != track_uri or current_round.lyrics_loaded:
                return
            try:
                media_item = await self.mass.music.get_item_by_uri(track_uri)
                lyrics, lrc_lyrics = (
                    await self.mass.metadata.get_track_lyrics(media_item)
                    if isinstance(media_item, Track)
                    else (None, None)
                )
            except Exception as err:
                self.logger.debug(
                    "Could not fetch Blind Test lyrics for %s: %s",
                    track_uri,
                    err,
                )
                lyrics, lrc_lyrics = None, None
            session = self._sessions.get(session_id)
            if (
                session is None
                or session.phase != BlindTestPhase.REVEAL
                or session.current_round_index != round_index
            ):
                return
            current_round = get_current_round(session)
            if current_round.track_uri != track_uri:
                return
            current_round.lyrics = lyrics
            current_round.lrc_lyrics = lrc_lyrics
            current_round.lyrics_loaded = True
            _touch_session(session)
        finally:
            self._lyrics_tasks.discard(task_key)

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


def _clean_session_name(name: str | None) -> str | None:
    """Return a normalized optional session name."""
    if not name:
        return None
    return name.strip() or None


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


def _touch_session(session: BlindTestSession) -> None:
    """Mark a session as updated."""
    session.updated_at = time.time()


def _session_summary(session: BlindTestSession) -> dict[str, Any]:
    """Return a compact host-visible live session summary."""
    current_round = (
        session.current_round_index + 1 if session.current_round_index is not None else 0
    )
    return {
        "session_id": session.session_id,
        "name": session.config.name or "Blind Test",
        "phase": session.phase,
        "player_count": len(session.players),
        "connected_player_count": sum(1 for player in session.players.values() if player.connected),
        "current_round": current_round,
        "round_count": session.config.round_count,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _session_info(session: BlindTestSession) -> dict[str, Any]:
    """Return public metadata for a joinable session."""
    return {
        "session_id": session.session_id,
        "name": session.config.name or "Blind Test",
        "phase": session.phase,
        "player_count": len(session.players),
        "round_count": session.config.round_count,
    }


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
            round_state.pop("lyrics_loaded", None)
            round_state.pop("lyrics", None)
            round_state.pop("lrc_lyrics", None)
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
        title=track.name,
    )
