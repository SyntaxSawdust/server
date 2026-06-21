"""
Streaming via a local go-librespot daemon for the Spotify provider.

One daemon runs per provider instance and plays a single track at a time, so
``stream_spotify_uri`` serializes concurrent requests (a second player or a queue
prefetch) on a lock. Each track is read off the daemon's stdout as fast as it is
produced (the daemon decodes ahead of realtime), so it buffers quickly and frees
the daemon for the next request. go-librespot keeps its output pipe open across
tracks, so end-of-track is detected from the ``stopped`` / ``not_playing``
WebSocket events rather than an stdout EOF.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import AudioError

from music_assistant.helpers.process import AsyncProcess
from music_assistant.helpers.util import select_free_port
from music_assistant.providers.spotify_connect.client import GoLibrespotClient
from music_assistant.providers.spotify_connect.helpers import (
    generate_device_id,
    get_go_librespot_binary,
)

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from .provider import SpotifyProvider

API_PORT_RANGE_START = 38900
API_PORT_RANGE_END = 39000
STREAM_READ_CHUNK = 16384
# how long a single stdout read waits before we re-check track-end / startup
READ_TIMEOUT_S = 0.3
# give up if the daemon produces no audio at all this long after a play request
STARTUP_TIMEOUT_S = 20.0
# safety net: end the stream if audio stops without a track-end event arriving
POST_STREAM_IDLE_TIMEOUT_S = 5.0
MAX_DAEMON_RESTARTS = 5


class GoLibrespotStreamer:
    """Owns a go-librespot daemon and streams single Spotify tracks through it."""

    def __init__(self, provider: SpotifyProvider) -> None:
        """Initialize the streamer for the given provider instance."""
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger
        self.cache_dir = provider.cache_dir
        self._binary: str | None = None
        self._api_port: int = 0
        self._client: GoLibrespotClient | None = None
        self._proc: AsyncProcess | None = None
        self._daemon_task: asyncio.Task[None] | None = None
        self._events_task: asyncio.Task[None] | None = None
        self._stop_called = False
        self._restart_error_count = 0
        # the single daemon plays one track at a time, so concurrent fetches
        # (a second player / queue prefetch) serialize here.
        self._fetch_lock = asyncio.Lock()
        # set by the events websocket when the current track finishes.
        self._track_ended = asyncio.Event()

    async def start(self) -> None:
        """Resolve the binary, pick a port and launch the supervised daemon + events tasks."""
        self._binary = get_go_librespot_binary()
        self._api_port = await select_free_port(API_PORT_RANGE_START, API_PORT_RANGE_END)
        self._client = GoLibrespotClient(
            self.mass, f"http://127.0.0.1:{self._api_port}", self.logger
        )
        self._daemon_task = self.mass.create_task(self._daemon_runner())
        self._events_task = self.mass.create_task(self._events_runner())

    async def stop(self) -> None:
        """Stop the daemon and events tasks."""
        self._stop_called = True
        for task in (self._events_task, self._daemon_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

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
        if self._client is None or self._proc is None:
            raise AudioError("go-librespot daemon is not running")
        async with self._fetch_lock:
            client = self._client
            proc = self._proc
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

    def _write_config(self, username: str, access_token: str) -> None:
        """
        Write the go-librespot ``config.yml`` for this instance.

        JSON is valid YAML, so we emit JSON to avoid an extra dependency. The
        daemon authenticates with a Spotify access token (no zeroconf device),
        decodes to PCM on its stdout, and stops at the end of a single-track
        context instead of rolling into autoplay/radio.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        config: dict[str, Any] = {
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
                "spotify_token": {"username": username, "access_token": access_token},
            },
            "server": {"enabled": True, "address": "127.0.0.1", "port": self._api_port},
        }
        config_file = os.path.join(self.cache_dir, "config.yml")
        with open(config_file, "w", encoding="utf-8") as fileobj:
            json.dump(config, fileobj, indent=2)

    async def _daemon_runner(self) -> None:
        """Run and supervise the go-librespot daemon, restarting it if it exits."""
        assert self._binary
        while True:
            proc: AsyncProcess | None = None
            try:
                # refresh the token on every (re)start: the daemon bootstraps its
                # Spotify session from it. _sp_user is populated by login() during
                # provider init, before the streamer is started.
                auth_info = await self.provider._get_auth_info()
                username = str(self.provider._sp_user["id"]) if self.provider._sp_user else ""
                self._write_config(username, auth_info["access_token"])
                # stdout carries the decoded PCM consumed by stream_spotify_uri;
                # stderr carries the daemon logs read here.
                self._proc = proc = AsyncProcess(
                    [self._binary, "--config_dir", self.cache_dir],
                    stdout=True,
                    stderr=True,
                    name=f"go-librespot[spotify:{self.provider.instance_id}]",
                )
                await proc.start()
                self.logger.info("Started Spotify go-librespot daemon")
                async for line in proc.iter_stderr():
                    self.logger.debug("[go-librespot] %s", line)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                self.logger.warning("go-librespot daemon error: %s", err)
            finally:
                if proc:
                    await proc.close()
                self._proc = None
            if self._stop_called:
                break
            self._restart_error_count += 1
            if self._restart_error_count >= MAX_DAEMON_RESTARTS:
                self.provider.unload_with_error("go-librespot daemon failed to start repeatedly.")
                return
            await asyncio.sleep(2)

    async def _events_runner(self) -> None:
        """Keep the events websocket connected and flag track-end events."""
        assert self._client is not None
        while not self._stop_called:
            try:
                if not await self._client.wait_until_ready():
                    await asyncio.sleep(2)
                    continue
                self._restart_error_count = 0
                await self._client.listen_events(self._handle_event)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                self.logger.debug("go-librespot events websocket dropped: %s", err)
            if not self._stop_called:
                await asyncio.sleep(2)

    async def _handle_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Flag the current fetch as done when the track finishes."""
        # 'not_playing' = the current track finished; 'stopped' = the (single-track)
        # context is empty. Either marks the end of the track we are streaming.
        if event_type in ("not_playing", "stopped"):
            self._track_ended.set()
