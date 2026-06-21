"""
Supervised go-librespot daemon manager.

Wraps the lifecycle shared by every go-librespot use: resolving the binary,
picking a loopback API port, (re)writing ``config.yml``, running the process and
restarting it on exit, and keeping the ``/events`` websocket connected. The
``GoLibrespotClient`` (REST) and the live process handle are exposed for callers
to drive playback and read the decoded PCM from the daemon's stdout.

The per-use specifics — the config contents and what to do with websocket events —
are supplied by the caller via ``config_builder`` and ``on_event``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from music_assistant.helpers.process import AsyncProcess
from music_assistant.helpers.util import select_free_port

from .client import EventCallback, GoLibrespotClient
from .helpers import get_go_librespot_binary

if TYPE_CHECKING:
    import logging

    from music_assistant.mass import MusicAssistant

API_PORT_RANGE_START = 38800
API_PORT_RANGE_END = 39000
MAX_RESTARTS = 5

# Called with the daemon's API port; returns the config.yml contents. Awaited on
# every (re)start so callers can refresh short-lived values such as access tokens.
ConfigBuilder = Callable[[int], Awaitable[dict[str, Any]]]


class GoLibrespotDaemon:
    """Runs and supervises a go-librespot daemon plus its events websocket."""

    def __init__(
        self,
        mass: MusicAssistant,
        *,
        cache_dir: str,
        logger: logging.Logger,
        config_builder: ConfigBuilder,
        on_event: EventCallback,
        name: str = "go-librespot",
        on_exit: Callable[[], None] | None = None,
        on_repeated_failure: Callable[[str], None] | None = None,
    ) -> None:
        """
        Initialize the daemon manager.

        :param mass: The MusicAssistant instance (for its loop / HTTP session).
        :param cache_dir: Directory holding ``config.yml`` and the credential cache.
        :param logger: Logger for diagnostics.
        :param config_builder: Async callable returning the ``config.yml`` dict for a given API port.
        :param on_event: Coroutine handling each ``/events`` websocket event.
        :param name: Short label used in logs and the process name.
        :param on_exit: Optional callback invoked after the daemon process exits (e.g. to reset state).
        :param on_repeated_failure: Optional callback invoked when the daemon fails to stay up.
        """
        self.mass = mass
        self.cache_dir = cache_dir
        self.logger = logger
        self.name = name
        self._config_builder = config_builder
        self._on_event = on_event
        self._on_exit = on_exit
        self._on_repeated_failure = on_repeated_failure
        self._binary: str | None = None
        self.api_port: int = 0
        self.client: GoLibrespotClient | None = None
        self.proc: AsyncProcess | None = None
        self._daemon_task: asyncio.Task[None] | None = None
        self._events_task: asyncio.Task[None] | None = None
        self._stop_called = False
        self._restart_error_count = 0

    async def start(self) -> None:
        """Resolve the binary, pick a port and launch the supervised daemon + events tasks."""
        self._binary = get_go_librespot_binary()
        self.api_port = await select_free_port(API_PORT_RANGE_START, API_PORT_RANGE_END)
        self.client = GoLibrespotClient(self.mass, f"http://127.0.0.1:{self.api_port}", self.logger)
        # Two self-healing supervisors: one keeps the daemon process alive, the
        # other keeps the events websocket connected (reconnecting across daemon
        # restarts) and resets the restart backoff once the websocket is healthy.
        self._daemon_task = self.mass.create_task(self._daemon_runner())
        self._events_task = self.mass.create_task(self._events_runner())

    async def stop(self) -> None:
        """Cancel the daemon and events supervisor tasks."""
        self._stop_called = True
        for task in (self._events_task, self._daemon_task):
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    def _write_config(self, config: dict[str, Any]) -> None:
        """Write the daemon's ``config.yml`` (JSON is valid YAML)."""
        os.makedirs(self.cache_dir, exist_ok=True)
        config_file = os.path.join(self.cache_dir, "config.yml")
        with open(config_file, "w", encoding="utf-8") as fileobj:
            json.dump(config, fileobj, indent=2)

    async def _daemon_runner(self) -> None:
        """Run and supervise the go-librespot daemon, restarting it if it exits."""
        assert self._binary
        while True:
            self._write_config(await self._config_builder(self.api_port))
            proc: AsyncProcess | None = None
            try:
                # stdout carries the decoded PCM (consumed by callers); stderr
                # carries the daemon logs, read here. The process pipe owns stdout
                # from spawn, so go-librespot's non-blocking pipe open never fails
                # for lack of a reader.
                self.proc = proc = AsyncProcess(
                    [self._binary, "--config_dir", self.cache_dir],
                    stdout=True,
                    stderr=True,
                    name=self.name,
                )
                await proc.start()
                self.logger.info("Started go-librespot daemon [%s]", self.name)
                async for line in proc.iter_stderr():
                    self.logger.debug("[%s] %s", self.name, line)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                self.logger.warning("go-librespot daemon error [%s]: %s", self.name, err)
            finally:
                if proc:
                    await proc.close()
                self.proc = None
                if self._on_exit:
                    self._on_exit()
            if self._stop_called:
                break
            self._restart_error_count += 1
            if self._restart_error_count >= MAX_RESTARTS:
                if self._on_repeated_failure:
                    self._on_repeated_failure("go-librespot daemon failed to start multiple times.")
                return
            await asyncio.sleep(2)

    async def _events_runner(self) -> None:
        """Keep the events websocket connected, reconnecting across daemon restarts."""
        assert self.client is not None
        while not self._stop_called:
            try:
                if not await self.client.wait_until_ready():
                    await asyncio.sleep(2)
                    continue
                # A live websocket means the daemon is healthy: reset the backoff.
                self._restart_error_count = 0
                await self.client.listen_events(self._on_event)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                self.logger.debug("go-librespot events websocket dropped [%s]: %s", self.name, err)
            if not self._stop_called:
                await asyncio.sleep(2)
