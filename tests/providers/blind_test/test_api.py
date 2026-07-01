"""Tests for Blind Test provider API commands."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import EventType, MediaType
from music_assistant_models.errors import InvalidDataError
from music_assistant_models.media_items import (
    ItemMapping,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)

from music_assistant.providers.blind_test import BlindTestPlugin, get_config_entries
from music_assistant.providers.blind_test.models import (
    BlindTestConfig,
    BlindTestPhase,
    BlindTestRound,
    BlindTestSession,
    BlindTestSuggestion,
)


def _create_plugin() -> BlindTestPlugin:
    """Create a minimally configured Blind Test plugin for unit tests."""
    plugin = BlindTestPlugin.__new__(BlindTestPlugin)
    plugin.mass = MagicMock()

    def _schedule_task(
        target: Callable[..., Awaitable[Any]] | Awaitable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Task[Any]:
        kwargs.pop("task_id", None)
        kwargs.pop("abort_existing", None)
        eager_start = kwargs.pop("eager_start", True)
        coro = cast(
            "Coroutine[Any, Any, Any]",
            target(*args, **kwargs) if callable(target) else target,
        )
        loop = asyncio.get_running_loop()
        try:
            return asyncio.Task(coro, loop=loop, eager_start=eager_start)
        except TypeError:
            return loop.create_task(coro)  # pragma: no cover - Python < 3.12 fallback

    plugin.mass.create_task = MagicMock(side_effect=_schedule_task)
    plugin.mass.player_queues.play_media = AsyncMock()
    plugin.mass.player_queues.stop = AsyncMock()
    plugin.mass.players.play_media = AsyncMock()
    plugin.mass.players.cmd_set_members = AsyncMock()
    plugin.mass.music.get_item_by_uri = AsyncMock(side_effect=_get_source_item_by_uri)
    plugin.mass.metadata.get_image_url_for_item = AsyncMock(return_value=None)
    plugin.mass.metadata.get_track_lyrics = AsyncMock(return_value=(None, None))
    plugin.logger = MagicMock()
    plugin._sessions = {}
    plugin._lyrics_tasks = set()
    plugin._unregister_handles = []
    return plugin


def _session(phase: BlindTestPhase = BlindTestPhase.LOBBY) -> BlindTestSession:
    """Create a test session."""
    return BlindTestSession(
        session_id="session",
        join_code="join",
        phase=phase,
        config=BlindTestConfig(player_id="queue_1"),
    )


def _track(item_id: str, name: str, artist: str, duration: int = 180) -> Track:
    """Create a track model for API tests."""
    return Track(
        item_id=item_id,
        provider="library",
        name=name,
        duration=duration,
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain="library",
                provider_instance="library",
            )
        },
        artists=UniqueList(
            [
                ItemMapping(
                    item_id=f"artist_{item_id}",
                    provider="library",
                    media_type=MediaType.ARTIST,
                    name=artist,
                )
            ]
        ),
    )


def _playlist(item_id: str = "playlist") -> Playlist:
    """Create a playlist model for API tests."""
    return Playlist(
        item_id=item_id,
        provider="library",
        name="Playlist",
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain="library",
                provider_instance="library",
            )
        },
    )


async def _get_source_item_by_uri(uri: str) -> Track | Playlist:
    """Return a source item for a test URI."""
    item_id = uri.rsplit("/", 1)[-1]
    if "/playlist/" in uri:
        return _playlist(item_id)
    return _track(item_id, f"Track {item_id}", "Artist")


def _session_with_round(phase: BlindTestPhase) -> BlindTestSession:
    """Create a test session with one round."""
    session = _session(phase)
    session.current_round_index = 0
    session.rounds.append(
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
                BlindTestSuggestion(suggestion_id="wrong_1", label="Justice - D.A.N.C.E."),
            ],
        )
    )
    return session


def _round_payload(index: int = 1) -> dict[str, object]:
    """Create a valid API round payload."""
    return {
        "track_uri": f"library://track/{index}",
        "answer_label": "Daft Punk - One More Time",
        "duration": 180,
        "suggestions": [
            {
                "suggestion_id": "correct",
                "label": "Daft Punk - One More Time",
                "is_correct": True,
            },
            {
                "suggestion_id": "wrong_1",
                "label": "Justice - D.A.N.C.E.",
                "is_correct": False,
            },
            {
                "suggestion_id": "wrong_2",
                "label": "Cassius - 1999",
                "is_correct": False,
            },
            {
                "suggestion_id": "wrong_3",
                "label": "Modjo - Lady",
                "is_correct": False,
            },
        ],
    }


@pytest.mark.asyncio
async def test_get_config_entries_returns_empty_tuple() -> None:
    """Expose provider setup config entries for the Settings UI."""
    assert await get_config_entries(MagicMock()) == ()


@pytest.mark.asyncio
async def test_loaded_in_mass_registers_api_commands() -> None:
    """Register all v1 Blind Test API commands."""
    plugin = _create_plugin()
    mass = cast("MagicMock", plugin.mass)
    unregister = MagicMock()
    mass.register_api_command.return_value = unregister

    await plugin.loaded_in_mass()

    assert mass.register_api_command.call_count == 15
    assert [call.args[0] for call in mass.register_api_command.call_args_list] == [
        "blind_test/create",
        "blind_test/session",
        "blind_test/sessions",
        "blind_test/rename",
        "blind_test/info",
        "blind_test/join",
        "blind_test/state",
        "blind_test/start",
        "blind_test/prepare_round",
        "blind_test/answer",
        "blind_test/reveal",
        "blind_test/ready",
        "blind_test/next",
        "blind_test/reset",
        "blind_test/delete",
    ]


@pytest.mark.asyncio
async def test_unload_unregisters_api_commands() -> None:
    """Unregister API commands when the provider unloads."""
    plugin = _create_plugin()
    unregister = MagicMock()
    plugin._unregister_handles = [unregister]

    await plugin.unload()

    unregister.assert_called_once()
    assert plugin._unregister_handles == []


@pytest.mark.asyncio
async def test_create_session_validates_min_round_count() -> None:
    """Reject a game with fewer than two rounds."""
    plugin = _create_plugin()

    with pytest.raises(InvalidDataError, match="at least 2 rounds"):
        await plugin.create_session(player_id="queue_1", round_count=1)


@pytest.mark.asyncio
async def test_create_session_returns_host_state() -> None:
    """Create a session and return host-visible state."""
    plugin = _create_plugin()

    state = await plugin.create_session(
        player_id="queue_1",
        round_count=3,
        suggestion_count=4,
        answer_duration=20,
        source_uris=["library://track/1"],
        name="Friday quiz",
    )

    assert state["session_id"] in plugin._sessions
    assert state["phase"] == BlindTestPhase.LOBBY
    assert state["config"]["round_count"] == 3
    assert state["config"]["name"] == "Friday quiz"
    assert state["config"]["play_on_joined_players"] is True
    assert state["sources"] == [
        {
            "uri": "library://track/1",
            "name": "Track 1",
            "media_type": "track",
        }
    ]
    assert state["created_at"] > 0
    assert state["updated_at"] > 0


@pytest.mark.asyncio
async def test_create_session_trims_optional_name() -> None:
    """Session names are trimmed and may be omitted."""
    plugin = _create_plugin()

    named = await plugin.create_session(
        player_id="queue_1",
        source_uris=["library://track/1"],
        name="  Friday quiz  ",
    )
    unnamed = await plugin.create_session(
        player_id="queue_1",
        source_uris=["library://track/1"],
        name="   ",
    )

    assert named["config"]["name"] == "Friday quiz"
    assert unnamed["config"]["name"] is None


@pytest.mark.asyncio
async def test_rename_session_updates_name_and_summary() -> None:
    """Rename an existing session and expose it in host summaries."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()

    renamed = await plugin.rename_session("session", "  Saturday quiz  ")

    assert renamed["config"]["name"] == "Saturday quiz"
    assert plugin._sessions["session"].updated_at > 0
    assert (await plugin.list_sessions())[0]["name"] == "Saturday quiz"

    unnamed = await plugin.rename_session("session", "   ")

    assert unnamed["config"]["name"] is None
    assert (await plugin.list_sessions())[0]["name"] == "Blind Test"


@pytest.mark.asyncio
async def test_get_session_info_returns_public_join_metadata() -> None:
    """Expose minimal session metadata to unauthenticated join pages."""
    plugin = _create_plugin()
    session = _session()
    session.config.name = "Saturday quiz"
    plugin._sessions["session"] = session
    await plugin.join_session("session", "Alice")

    info = await plugin.get_session_info("session")

    assert info == {
        "session_id": "session",
        "name": "Saturday quiz",
        "phase": BlindTestPhase.LOBBY,
        "player_count": 1,
        "round_count": 2,
    }


@pytest.mark.asyncio
async def test_list_sessions_returns_live_session_summaries() -> None:
    """Return compact host summaries for the active Blind Test rooms."""
    plugin = _create_plugin()
    first = await plugin.create_session(
        player_id="queue_1",
        source_uris=["library://track/1"],
        name="First room",
    )
    second = await plugin.create_session(
        player_id="queue_1",
        source_uris=["library://track/2"],
        name="Second room",
    )
    await plugin.join_session(first["session_id"], "Alice")

    summaries = await plugin.list_sessions()

    assert [summary["session_id"] for summary in summaries] == [
        first["session_id"],
        second["session_id"],
    ]
    assert summaries[0] == {
        "session_id": first["session_id"],
        "name": "First room",
        "phase": BlindTestPhase.LOBBY,
        "player_count": 1,
        "connected_player_count": 1,
        "current_round": 0,
        "round_count": 2,
        "created_at": first["created_at"],
        "updated_at": plugin._sessions[first["session_id"]].updated_at,
    }


@pytest.mark.asyncio
async def test_join_session_returns_player_token_and_player_state() -> None:
    """Join a session as a player."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()

    result = await plugin.join_session("session", "Alice", "blind_test_session_alice")

    assert result["player_token"]
    assert result["player_id"] in plugin._sessions["session"].players
    assert result["state"]["current_player_id"] == result["player_id"]
    assert result["state"]["players"][result["player_id"]]["name"] == "Alice"
    assert (
        result["state"]["players"][result["player_id"]]["sendspin_player_id"]
        == "blind_test_session_alice"
    )
    assert "token_hash" not in result["state"]["players"][result["player_id"]]


@pytest.mark.asyncio
async def test_get_player_state_updates_temporary_sendspin_player_id() -> None:
    """Reconnect polling can refresh the temporary Sendspin phone player ID."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    joined = await plugin.join_session("session", "Alice")

    state = await plugin.get_player_state(
        "session",
        joined["player_token"],
        "blind_test_session_alice",
    )

    assert state["players"][joined["player_id"]]["sendspin_player_id"] == "blind_test_session_alice"


@pytest.mark.asyncio
async def test_late_joiner_is_grouped_with_current_selected_player() -> None:
    """Late joined phones are added to the active synced playback group."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload())
    mass = cast("MagicMock", plugin.mass)
    mass.players.get_player.return_value = SimpleNamespace(
        can_group_with=["blind_test_session_bob"]
    )

    result = await plugin.join_session("session", "Bob", "blind_test_session_bob")

    assert result["state"]["players"][result["player_id"]]["active_from_round"] == 1
    mass.players.cmd_set_members.assert_awaited_with(
        target_player="queue_1",
        player_ids_to_add=["blind_test_session_bob"],
    )


@pytest.mark.asyncio
async def test_late_reconnect_with_sendspin_id_is_grouped() -> None:
    """A late player that registers Sendspin after joining is grouped on poll."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload())
    joined = await plugin.join_session("session", "Bob")
    mass = cast("MagicMock", plugin.mass)
    mass.players.get_player.return_value = SimpleNamespace(
        can_group_with=["blind_test_session_bob"]
    )

    await plugin.get_player_state(
        "session",
        joined["player_token"],
        "blind_test_session_bob",
    )

    mass.players.cmd_set_members.assert_awaited_with(
        target_player="queue_1",
        player_ids_to_add=["blind_test_session_bob"],
    )


@pytest.mark.asyncio
async def test_repeated_state_poll_does_not_regroup_same_sendspin_id() -> None:
    """Polling state with the same Sendspin ID should not restart grouped playback."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload())
    joined = await plugin.join_session("session", "Bob")
    mass = cast("MagicMock", plugin.mass)
    mass.players.get_player.return_value = SimpleNamespace(
        can_group_with=["blind_test_session_bob"]
    )

    await plugin.get_player_state(
        "session",
        joined["player_token"],
        "blind_test_session_bob",
    )
    await plugin.get_player_state(
        "session",
        joined["player_token"],
        "blind_test_session_bob",
    )

    mass.players.cmd_set_members.assert_awaited_once_with(
        target_player="queue_1",
        player_ids_to_add=["blind_test_session_bob"],
    )


@pytest.mark.asyncio
async def test_join_session_rejects_duplicate_names() -> None:
    """Reject duplicate names in the same session."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice")

    with pytest.raises(InvalidDataError, match="unique"):
        await plugin.join_session("session", "alice")


@pytest.mark.asyncio
async def test_join_session_late_player_starts_next_round() -> None:
    """Late joiners become active from the next round."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session_with_round(BlindTestPhase.ANSWERING)

    result = await plugin.join_session("session", "Alice")

    player = plugin._sessions["session"].players[result["player_id"]]
    assert player.active_from_round == 1


@pytest.mark.asyncio
async def test_get_player_state_hides_correct_answer_while_answering() -> None:
    """Do not leak the correct answer during the answering phase."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session_with_round(BlindTestPhase.ANSWERING)
    joined = await plugin.join_session("session", "Alice")

    state = await plugin.get_player_state("session", joined["player_token"])

    assert "is_correct" not in state["rounds"][0]["suggestions"][0]
    assert "is_correct" not in state["rounds"][0]["suggestions"][1]


@pytest.mark.asyncio
async def test_get_player_state_hides_lyrics_while_answering() -> None:
    """Do not leak reveal lyrics during the answering phase."""
    plugin = _create_plugin()
    session = _session_with_round(BlindTestPhase.ANSWERING)
    session.rounds[0].lyrics = "plain lyrics"
    session.rounds[0].lrc_lyrics = "[00:01.00]synced lyrics"
    session.rounds[0].lyrics_loaded = True
    plugin._sessions["session"] = session
    joined = await plugin.join_session("session", "Alice")

    state = await plugin.get_player_state("session", joined["player_token"])

    assert "lyrics" not in state["rounds"][0]
    assert "lrc_lyrics" not in state["rounds"][0]
    assert "lyrics_loaded" not in state["rounds"][0]


@pytest.mark.asyncio
async def test_get_player_state_reveals_correct_answer_during_reveal() -> None:
    """Expose the correct answer once the session is in reveal."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session_with_round(BlindTestPhase.REVEAL)
    joined = await plugin.join_session("session", "Alice")

    state = await plugin.get_player_state("session", joined["player_token"])

    assert state["rounds"][0]["suggestions"][0]["is_correct"] is True
    assert state["rounds"][0]["suggestions"][1]["is_correct"] is False


@pytest.mark.asyncio
async def test_join_session_includes_reveal_lyrics() -> None:
    """Late joiners during reveal receive lyrics once the background task completes."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session_with_round(BlindTestPhase.REVEAL)
    mass = cast("MagicMock", plugin.mass)
    lyrics_requested = asyncio.Event()
    release_lyrics = asyncio.Event()

    async def delayed_lyrics(_track: Track) -> tuple[str, str]:
        lyrics_requested.set()
        await release_lyrics.wait()
        return ("plain lyrics", "[00:01.00]synced lyrics")

    mass.metadata.get_track_lyrics = AsyncMock(side_effect=delayed_lyrics)

    joined = await plugin.join_session("session", "Alice")

    assert joined["state"]["rounds"][0].get("lyrics") is None
    assert joined["state"]["rounds"][0]["lyrics_loaded"] is False
    await asyncio.wait_for(lyrics_requested.wait(), timeout=1)
    release_lyrics.set()
    await asyncio.sleep(0)
    state = await plugin.get_player_state("session", joined["player_token"])

    assert state["rounds"][0]["lyrics"] == "plain lyrics"
    assert state["rounds"][0]["lrc_lyrics"] == "[00:01.00]synced lyrics"
    assert state["rounds"][0]["lyrics_loaded"] is True
    mass.metadata.get_track_lyrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_reveal_does_not_wait_for_lyrics() -> None:
    """Answer responses reveal the round without waiting for lyrics metadata."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    mass = cast("MagicMock", plugin.mass)
    lyrics_requested = asyncio.Event()
    release_lyrics = asyncio.Event()

    async def delayed_lyrics(_track: Track) -> tuple[str, str]:
        lyrics_requested.set()
        await release_lyrics.wait()
        return ("plain lyrics", "[00:01.00]synced lyrics")

    mass.metadata.get_track_lyrics = AsyncMock(
        side_effect=delayed_lyrics,
    )
    joined = await plugin.join_session("session", "Alice")
    await plugin.start_session("session", _round_payload())

    state = await asyncio.wait_for(
        plugin.answer("session", joined["player_token"], "correct"),
        timeout=1,
    )

    assert state["phase"] == BlindTestPhase.REVEAL
    assert state["rounds"][0].get("lyrics") is None
    assert state["rounds"][0]["lyrics_loaded"] is False
    await asyncio.wait_for(lyrics_requested.wait(), timeout=1)
    release_lyrics.set()
    await asyncio.sleep(0)
    revealed = await plugin.get_player_state("session", joined["player_token"])
    assert revealed["rounds"][0]["lyrics"] == "plain lyrics"
    assert revealed["rounds"][0]["lyrics_loaded"] is True
    mass.metadata.get_track_lyrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_player_state_rejects_unknown_token() -> None:
    """Reject state requests without a valid player token."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()

    with pytest.raises(InvalidDataError, match="Unknown Blind Test player"):
        await plugin.get_player_state("session", "missing")


@pytest.mark.asyncio
async def test_start_session_starts_first_round_and_playback() -> None:
    """Start a round and hand playback to the configured player queue."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()

    state = await plugin.start_session("session", _round_payload())

    assert state["phase"] == BlindTestPhase.ANSWERING
    assert state["current_round_index"] == 0
    assert state["rounds"][0]["started_at"] is not None
    mass = cast("MagicMock", plugin.mass)
    assert mass.player_queues.play_media.await_args_list[0].kwargs == {
        "queue_id": "queue_1",
        "media": "library://track/1",
    }


@pytest.mark.asyncio
async def test_start_session_plays_round_on_joined_phone_players() -> None:
    """Phone Sendspin players are played through their queue for stream resolution."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")

    await plugin.start_session("session", _round_payload())

    mass = cast("MagicMock", plugin.mass)
    assert mass.player_queues.play_media.await_args_list[1].kwargs == {
        "queue_id": "blind_test_session_alice",
        "media": "library://track/1",
    }


@pytest.mark.asyncio
async def test_start_session_can_play_only_on_joined_phone_players() -> None:
    """Host can create a phone-only session without queueing to the selected player."""
    plugin = _create_plugin()
    state = await plugin.create_session(
        player_id="",
        source_uris=["library://track/1"],
        play_on_player=False,
        play_on_joined_players=True,
    )
    await plugin.join_session(state["session_id"], "Alice", "blind_test_session_alice")

    await plugin.start_session(state["session_id"], _round_payload())

    mass = cast("MagicMock", plugin.mass)
    mass.player_queues.play_media.assert_awaited_once_with(
        queue_id="blind_test_session_alice",
        media="library://track/1",
    )


@pytest.mark.asyncio
async def test_start_session_ignores_unavailable_joined_phone_player() -> None:
    """A stale temporary phone player should not prevent the round from starting."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    mass = cast("MagicMock", plugin.mass)
    mass.player_queues.play_media.side_effect = [None, RuntimeError("missing")]

    state = await plugin.start_session("session", _round_payload())

    assert state["phase"] == BlindTestPhase.ANSWERING
    cast("MagicMock", plugin.logger.warning).assert_called_once()


@pytest.mark.asyncio
async def test_start_session_still_plays_phone_when_selected_player_fails() -> None:
    """A failing selected player should not prevent joined phones from playing."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    mass = cast("MagicMock", plugin.mass)
    mass.player_queues.play_media.side_effect = [RuntimeError("airplay failed"), None]

    state = await plugin.start_session("session", _round_payload())

    assert state["phase"] == BlindTestPhase.ANSWERING
    assert [call.kwargs["queue_id"] for call in mass.player_queues.play_media.await_args_list] == [
        "queue_1",
        "blind_test_session_alice",
    ]
    cast("MagicMock", plugin.logger.warning).assert_called_once()


@pytest.mark.asyncio
async def test_start_session_groups_compatible_phone_with_selected_player() -> None:
    """Compatible Sendspin targets are grouped so playback starts in sync."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    mass = cast("MagicMock", plugin.mass)
    mass.players.get_player.return_value = SimpleNamespace(
        can_group_with=["blind_test_session_alice"]
    )

    await plugin.start_session("session", _round_payload())

    mass.players.cmd_set_members.assert_awaited_once_with(
        target_player="queue_1",
        player_ids_to_add=["blind_test_session_alice"],
    )
    mass.player_queues.play_media.assert_awaited_once_with(
        queue_id="queue_1",
        media="library://track/1",
    )


@pytest.mark.asyncio
async def test_start_session_groups_provider_compatible_sendspin_phone() -> None:
    """Sendspin phones can group through provider-level compatibility."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    mass = cast("MagicMock", plugin.mass)

    def get_player(player_id: str) -> SimpleNamespace:
        if player_id == "queue_1":
            return SimpleNamespace(can_group_with=["sendspin"])
        return SimpleNamespace(provider=SimpleNamespace(instance_id="sendspin"))

    mass.players.get_player.side_effect = get_player

    await plugin.start_session("session", _round_payload())

    mass.players.cmd_set_members.assert_awaited_once_with(
        target_player="queue_1",
        player_ids_to_add=["blind_test_session_alice"],
    )
    mass.player_queues.play_media.assert_awaited_once_with(
        queue_id="queue_1",
        media="library://track/1",
    )


@pytest.mark.asyncio
async def test_start_session_groups_compatible_phone_only_players() -> None:
    """Phone-only sessions use one synced Sendspin leader when possible."""
    plugin = _create_plugin()
    state = await plugin.create_session(
        player_id="",
        source_uris=["library://track/1"],
        play_on_player=False,
        play_on_joined_players=True,
    )
    await plugin.join_session(state["session_id"], "Alice", "blind_test_session_alice")
    await plugin.join_session(state["session_id"], "Bob", "blind_test_session_bob")
    mass = cast("MagicMock", plugin.mass)
    mass.players.get_player.return_value = SimpleNamespace(
        can_group_with=["blind_test_session_bob"]
    )

    await plugin.start_session(state["session_id"], _round_payload())

    mass.players.cmd_set_members.assert_awaited_once_with(
        target_player="blind_test_session_alice",
        player_ids_to_add=["blind_test_session_bob"],
    )
    mass.player_queues.play_media.assert_awaited_once_with(
        queue_id="blind_test_session_alice",
        media="library://track/1",
    )


@pytest.mark.asyncio
async def test_start_session_ignores_stale_temporary_selected_player() -> None:
    """Temporary Blind Test players are not treated as stable selected outputs."""
    plugin = _create_plugin()
    session = _session()
    session.config.player_id = "blind_test_old_session_phone"
    plugin._sessions["session"] = session
    await plugin.join_session("session", "Alice", "blind_test_session_alice")

    await plugin.start_session("session", _round_payload())

    mass = cast("MagicMock", plugin.mass)
    mass.player_queues.play_media.assert_awaited_once_with(
        queue_id="blind_test_session_alice",
        media="library://track/1",
    )
    cast("MagicMock", plugin.logger.warning).assert_any_call(
        "Ignoring stale temporary Blind Test player %s as selected playback output",
        "blind_test_old_session_phone",
    )


@pytest.mark.asyncio
async def test_prepare_round_builds_suggestions_from_configured_source() -> None:
    """Prepare a playable round from a configured track and search distractors."""
    plugin = _create_plugin()
    session = _session()
    session.config.source_uris = ["library://track/source"]
    plugin._sessions["session"] = session
    mass = cast("MagicMock", plugin.mass)
    mass.music.get_item_by_uri = AsyncMock(
        return_value=_track("source", "One More Time", "Daft Punk")
    )
    mass.music.search = AsyncMock(
        return_value=SearchResults(
            tracks=[
                _track("wrong_1", "D.A.N.C.E.", "Justice"),
                _track("wrong_2", "1999", "Cassius"),
                _track("wrong_3", "Lady", "Modjo"),
            ]
        )
    )
    mass.metadata.get_image_url_for_item = AsyncMock(return_value="http://image/cover.jpg")

    payload = await plugin.prepare_round("session")

    assert payload["track_uri"] == "library://track/source"
    assert payload["answer_label"] == "Daft Punk - One More Time"
    assert payload["image_url"] == "http://image/cover.jpg"
    assert payload["duration"] == 180
    assert len(payload["suggestions"]) == 4
    assert sum(1 for item in payload["suggestions"] if item["is_correct"]) == 1


@pytest.mark.asyncio
async def test_get_next_source_track_randomizes_playlist_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pick from all unused playlist tracks instead of always returning the first track."""
    plugin = _create_plugin()
    session = _session()
    session.config.source_uris = ["library://playlist/playlist"]
    plugin._sessions["session"] = session
    mass = cast("MagicMock", plugin.mass)
    mass.music.get_item_by_uri = AsyncMock(return_value=_playlist())

    async def playlist_tracks(
        item_id: str,
        provider_instance_id_or_domain: str,
    ) -> AsyncGenerator[Track]:
        assert item_id == "playlist"
        assert provider_instance_id_or_domain == "library"
        yield _track("first", "First", "Artist")
        yield _track("second", "Second", "Artist")

    mass.music.playlists.tracks = playlist_tracks
    monkeypatch.setattr(
        "music_assistant.providers.blind_test.secrets.choice",
        lambda items: cast("list[Any]", items)[-1],
    )

    track = await plugin._get_next_source_track(session)

    assert track.item_id == "second"


@pytest.mark.asyncio
async def test_start_session_without_sources_is_rejected() -> None:
    """A prepared round needs at least one configured source URI."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()

    with pytest.raises(InvalidDataError, match="source URI"):
        await plugin.start_session("session")


@pytest.mark.asyncio
async def test_answer_locks_player_choice() -> None:
    """Players can answer once and cannot change their selection."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    joined = await plugin.join_session("session", "Alice")
    await plugin.join_session("session", "Bob")
    await plugin.start_session("session", _round_payload())

    state = await plugin.answer("session", joined["player_token"], "correct")

    assert state["rounds"][0]["answers"][joined["player_id"]]["suggestion_id"] == "correct"
    with pytest.raises(InvalidDataError, match="already answered"):
        await plugin.answer("session", joined["player_token"], "wrong_1")


@pytest.mark.asyncio
async def test_reveal_scores_correct_answers() -> None:
    """Reveal applies linear scoring to correct answers."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    alice = await plugin.join_session("session", "Alice")
    bob = await plugin.join_session("session", "Bob")
    await plugin.start_session("session", _round_payload())
    await plugin.answer("session", alice["player_token"], "correct")
    state = await plugin.answer("session", bob["player_token"], "wrong_1")

    assert state["phase"] == BlindTestPhase.REVEAL
    assert state["players"][alice["player_id"]]["score"] == 1000
    assert state["players"][bob["player_id"]]["score"] == 0
    assert state["rounds"][0]["answers"][alice["player_id"]]["points"] == 1000


@pytest.mark.asyncio
async def test_state_reveals_round_when_answer_duration_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling moves a no-answer round to reveal when time is up."""
    plugin = _create_plugin()
    session = _session()
    session.config.answer_duration = 30
    plugin._sessions["session"] = session
    await plugin.start_session("session", _round_payload())
    session.rounds[0].started_at = 10
    monkeypatch.setattr("music_assistant.providers.blind_test.time.time", lambda: 41)

    state = await plugin.get_session("session")

    assert state["phase"] == BlindTestPhase.REVEAL
    assert state["rounds"][0]["answers"] == {}


@pytest.mark.asyncio
async def test_state_reveals_round_when_track_duration_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling caps answer time at the current track duration."""
    plugin = _create_plugin()
    session = _session()
    session.config.answer_duration = 30
    plugin._sessions["session"] = session
    await plugin.start_session("session", {**_round_payload(), "duration": 12})
    session.rounds[0].started_at = 10
    monkeypatch.setattr("music_assistant.providers.blind_test.time.time", lambda: 22)

    state = await plugin.get_session("session")

    assert state["phase"] == BlindTestPhase.REVEAL


@pytest.mark.asyncio
async def test_unknown_track_duration_uses_answer_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown track length falls back to the configured answer duration."""
    plugin = _create_plugin()
    session = _session()
    session.config.answer_duration = 30
    plugin._sessions["session"] = session
    await plugin.start_session("session", {**_round_payload(), "duration": None})
    session.rounds[0].started_at = 10
    monkeypatch.setattr("music_assistant.providers.blind_test.time.time", lambda: 22)

    state = await plugin.get_session("session")

    assert state["phase"] == BlindTestPhase.ANSWERING


@pytest.mark.asyncio
async def test_answer_reveals_round_when_all_active_players_answered() -> None:
    """The last active answer moves players to the reveal screen."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    alice = await plugin.join_session("session", "Alice")
    bob = await plugin.join_session("session", "Bob")
    await plugin.start_session("session", _round_payload())
    await plugin.answer("session", alice["player_token"], "wrong_1")

    state = await plugin.answer("session", bob["player_token"], "correct")

    assert state["phase"] == BlindTestPhase.REVEAL
    assert state["rounds"][0]["answers"][bob["player_id"]]["points"] == 1000


@pytest.mark.asyncio
async def test_ready_advances_when_all_active_players_are_ready() -> None:
    """The last ready player advances the game to the next round."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    plugin.prepare_round = AsyncMock(return_value=_round_payload(2))  # type: ignore[method-assign]
    alice = await plugin.join_session("session", "Alice")
    bob = await plugin.join_session("session", "Bob")
    await plugin.start_session("session", _round_payload())
    await plugin.answer("session", alice["player_token"], "correct")
    await plugin.reveal("session")

    alice_state = await plugin.ready("session", alice["player_token"])
    bob_state = await plugin.ready("session", bob["player_token"])

    assert alice_state["phase"] == BlindTestPhase.REVEAL
    assert bob_state["phase"] == BlindTestPhase.ANSWERING
    assert bob_state["current_round_index"] == 1


@pytest.mark.asyncio
async def test_ready_finishes_when_last_round_is_ready() -> None:
    """The last ready player finishes the game after the final round."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    joined = await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload(1))
    await plugin.answer("session", joined["player_token"], "correct")
    await plugin.next_round("session", _round_payload(2))
    await plugin.answer("session", joined["player_token"], "correct")

    state = await plugin.ready("session", joined["player_token"])

    assert state["phase"] == BlindTestPhase.FINISHED
    mass = cast("MagicMock", plugin.mass)
    assert [call.args[0] for call in mass.player_queues.stop.await_args_list] == [
        "blind_test_session_alice",
        "queue_1",
    ]


@pytest.mark.asyncio
async def test_next_round_advances_then_finishes() -> None:
    """Host can advance after reveal and finish after the configured rounds."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    joined = await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload(1))
    await plugin.answer("session", joined["player_token"], "correct")

    second_round = await plugin.next_round("session", _round_payload(2))
    await plugin.answer("session", joined["player_token"], "correct")
    finished = await plugin.next_round("session")

    assert second_round["phase"] == BlindTestPhase.ANSWERING
    assert second_round["current_round_index"] == 1
    assert finished["phase"] == BlindTestPhase.FINISHED
    mass = cast("MagicMock", plugin.mass)
    assert [call.args[0] for call in mass.player_queues.stop.await_args_list] == [
        "blind_test_session_alice",
        "queue_1",
    ]


@pytest.mark.asyncio
async def test_reset_session_keeps_room_and_clears_game_state() -> None:
    """Host can start a fresh game with the same settings and players."""
    plugin = _create_plugin()
    session = _session()
    session.config.source_uris = ["library://track/1"]
    plugin._sessions["session"] = session
    joined = await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload(1))
    await plugin.answer("session", joined["player_token"], "correct")

    state = await plugin.reset_session("session")

    assert state["session_id"] == "session"
    assert state["join_code"] == "join"
    assert state["phase"] == BlindTestPhase.LOBBY
    assert state["rounds"] == []
    assert state["current_round_index"] is None
    assert state["config"]["source_uris"] == ["library://track/1"]
    assert state["players"][joined["player_id"]]["score"] == 0
    assert state["players"][joined["player_id"]]["ready"] is False
    assert state["players"][joined["player_id"]]["active_from_round"] == 0
    mass = cast("MagicMock", plugin.mass)
    assert [call.args[0] for call in mass.player_queues.stop.await_args_list] == [
        "blind_test_session_alice",
        "queue_1",
    ]


@pytest.mark.asyncio
async def test_delete_session_stops_playback_and_removes_room() -> None:
    """Host can close a Blind Test room from the live sessions list."""
    plugin = _create_plugin()
    plugin._sessions["session"] = _session()
    await plugin.join_session("session", "Alice", "blind_test_session_alice")
    await plugin.start_session("session", _round_payload(1))

    result = await plugin.delete_session("session")

    assert result == {"session_id": "session"}
    assert "session" not in plugin._sessions
    mass = cast("MagicMock", plugin.mass)
    assert [call.args[0] for call in mass.player_queues.stop.await_args_list] == [
        "blind_test_session_alice",
        "queue_1",
    ]
    mass.signal_event.assert_called_once_with(
        EventType.UNKNOWN,
        object_id="session",
        data={"type": "blind_test_session_removed", "session_id": "session"},
    )
