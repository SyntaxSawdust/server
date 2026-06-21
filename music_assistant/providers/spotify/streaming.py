"""
Streaming via a local go-librespot daemon for the Spotify provider.

A single daemon (managed by the shared ``GoLibrespotDaemon`` from the Spotify
Connect plugin) runs per provider instance and plays one track at a time, so
``stream_spotify_uri`` serializes concurrent requests (a second player or a queue
prefetch) on a lock. Each track is read off the daemon's stdout as fast as it is
produced (the daemon decodes ahead of realtime), so it buffers quickly and frees
the daemon for the next request. go-librespot keeps its output pipe open across
tracks, so end-of-track is detected from the ``stopped`` / ``not_playing``
websocket events rather than an stdout EOF.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import AudioError

from music_assistant.providers.spotify_connect.daemon import GoLibrespotDaemon
from music_assistant.providers.spotify_connect.helpers import generate_device_id

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from .provider import SpotifyProvider

STREAM_READ_CHUNK = 16384
# how long a single stdout read waits before we re-check track-end / startup
READ_TIMEOUT_S = 0.3
# give up if the daemon produces no audio at all this long after a play request
STARTUP_TIMEOUT_S = 20.0
# safety net: end the stream if audio stops without a track-end event arriving
POST_STREAM_IDLE_TIMEOUT_S = 5.0


class GoLibrespotStreamer:
    """Streams single Spotify tracks through a shared go-librespot daemon."""

    def __init__(self, provider: SpotifyProvider) -> None:
        """Initialize the streamer and its daemon for the given provider instance."""
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger
        self._daemon = GoLibrespotDaemon(
            provider.mass,
            cache_dir=provider.cache_dir,
            logger=provider.logger,
            config_builder=self._build_config,
            on_event=self._handle_event,
            name=f"go-librespot[spotify:{provider.instance_id}]",
            on_repeated_failure=provider.unload_with_error,
        )
        # the single daemon plays one track at a time, so concurrent fetches
        # (a second player / queue prefetch) serialize here.
        self._fetch_lock = asyncio.Lock()
        # set by the events websocket when the current track finishes.
        self._track_ended = asyncio.Event()

    async def start(self) -> None:
        """Launch the supervised go-librespot daemon."""
        await self._daemon.start()

    async def stop(self) -> None:
        """Stop the go-librespot daemon."""
        await self._daemon.stop()

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Stream a single track/episode for the given streamdetails."""
        media_type = "episode" if streamdetails.media_type == MediaType.PODCAST_EPISODE else "track"
        spotify_uri = f"spotify:{media_type}:{streamdetails.item_id}"
        async for chunk in self.stream_spotify_uri(spotify_uri, seek_position):
            yield chunk

    async def stream_spotify_uri(
        self, spotify_uri: str, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Play a single Spotify URI on the daemon and yield its decoded PCM."""
        client = self._daemon.client
        proc = self._daemon.proc
        if client is None or proc is None:
            raise AudioError("go-librespot daemon is not running")
        async with self._fetch_lock:
            self._track_ended.clear()
            await client.play(spotify_uri, paused=False)
            if seek_position:
                await client.seek(int(seek_position) * 1000)

            loop = self.mass.loop
            startup_deadline = loop.time() + STARTUP_TIMEOUT_S
            last_data = loop.time()
            yielded = False
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.read(STREAM_READ_CHUNK), timeout=READ_TIMEOUT_S
                    )
                except TimeoutError:
                    now = loop.time()
                    if self._track_ended.is_set():
                        return  # track finished and the pipe is drained
                    if not yielded:
                        if now > startup_deadline:
                            raise AudioError(
                                f"Timed out starting Spotify playback for {spotify_uri}"
                            )
                    elif now - last_data > POST_STREAM_IDLE_TIMEOUT_S:
                        # audio stopped without a track-end event; assume done
                        return
                    continue
                if not chunk:
                    return  # daemon stdout closed (process exited / restarting)
                yielded = True
                last_data = loop.time()
                yield chunk

    async def _build_config(self, api_port: int) -> dict[str, Any]:
        """
        Build the go-librespot ``config.yml`` for the daemon.

        The daemon authenticates with a Spotify access token (no zeroconf device),
        decodes to PCM on its stdout, and stops at the end of a single-track
        context instead of rolling into autoplay/radio. The token is refreshed on
        every (re)start because the daemon bootstraps its session from it.
        """
        auth_info = await self.provider._get_auth_info()
        username = str(self.provider._sp_user["id"]) if self.provider._sp_user else ""
        return {
            "device_name": "Music Assistant",
            "device_type": "speaker",
            "device_id": generate_device_id(self.provider.instance_id),
            "bitrate": 320,
            "audio_backend": "pipe",
            "audio_output_pipe": "/dev/stdout",
            "audio_output_pipe_format": "s16le",
            # don't let go-librespot attenuate the PCM; MA owns volume/conversion.
            "external_volume": True,
            # a bare track URI resolves to a one-track context; without this the
            # daemon would roll into Spotify radio at the end of the track.
            "disable_autoplay": True,
            "zeroconf_enabled": False,
            "credentials": {
                "type": "spotify_token",
                "spotify_token": {"username": username, "access_token": auth_info["access_token"]},
            },
            "server": {"enabled": True, "address": "127.0.0.1", "port": api_port},
        }

    async def _handle_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Flag the current fetch as done when the track finishes."""
        # 'not_playing' = the current track finished; 'stopped' = the (single-track)
        # context is empty. Either marks the end of the track we are streaming.
        if event_type in ("not_playing", "stopped"):
            self._track_ended.set()
