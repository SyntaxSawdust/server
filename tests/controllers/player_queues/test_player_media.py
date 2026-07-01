"""Tests for PlayerMedia creation from queue items."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.media_items import ItemMapping, ProviderMapping, Track, UniqueList
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData


def _track() -> Track:
    """Return a track with real metadata."""
    return Track(
        item_id="track_1",
        provider="library",
        name="One More Time",
        provider_mappings={
            ProviderMapping(
                item_id="track_1",
                provider_domain="library",
                provider_instance="library",
            )
        },
        artists=UniqueList(
            [
                ItemMapping(
                    item_id="artist_1",
                    provider="library",
                    media_type=MediaType.ARTIST,
                    name="Daft Punk",
                )
            ]
        ),
    )


def _controller(queue_id: str) -> PlayerQueuesController:
    """Return a minimal controller with one queue."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    ctrl.mass = MagicMock()
    ctrl._queue_data = {
        queue_id: PlayerQueueData(
            queue=PlayerQueue(
                queue_id=queue_id,
                active=True,
                display_name="Queue",
                available=True,
                items=1,
                session_id="session",
            )
        )
    }
    return cast("PlayerQueuesController", ctrl)


@pytest.mark.asyncio
async def test_player_media_masks_metadata_for_blind_test_queue() -> None:
    """Blind Test phone queues must not leak the answer through player metadata."""
    queue_id = "blind_test_session_phone"
    ctrl = _controller(queue_id)
    queue_item = QueueItem(
        queue_id=queue_id,
        queue_item_id="item_1",
        name="One More Time",
        duration=320,
        media_item=_track(),
    )

    media = await ctrl.player_media_from_queue_item(queue_item)

    assert media.title == "Blind Test"
    assert media.artist == "Music Assistant"
    assert media.album == ""


@pytest.mark.asyncio
async def test_player_media_keeps_metadata_for_regular_queue() -> None:
    """Normal queues keep their regular player metadata."""
    queue_id = "living_room"
    ctrl = _controller(queue_id)
    queue_item = QueueItem(
        queue_id=queue_id,
        queue_item_id="item_1",
        name="One More Time",
        duration=320,
        media_item=_track(),
    )

    media = await ctrl.player_media_from_queue_item(queue_item)

    assert media.title == "One More Time"
    assert media.artist == "Daft Punk"
