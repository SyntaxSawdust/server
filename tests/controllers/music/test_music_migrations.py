"""Tests for the music library database migrations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.errors import MusicAssistantError

from music_assistant.constants import DB_TABLE_PLAYLOG
from music_assistant.controllers.music.migrations import migrate_database
from music_assistant.helpers.database import DatabaseConnection

if TYPE_CHECKING:
    from pathlib import Path


async def test_migrate_database_rejects_too_old_schema() -> None:
    """Schema versions older than the minimum supported version are refused up-front."""
    create_tables = AsyncMock()
    with pytest.raises(MusicAssistantError):
        await migrate_database(
            MagicMock(),  # mass
            MagicMock(),  # database
            MagicMock(),  # logger
            prev_version=14,
            create_tables=create_tables,
        )
    # the guard fires before any schema work happens
    create_tables.assert_not_awaited()


async def _create_legacy_playlog_database(db_path: str) -> DatabaseConnection:
    """Create a database with the pre-v23 playlog table incl. later column migrations."""
    database = DatabaseConnection(db_path)
    await database.setup()
    # original playlog table with the legacy 3-column table-level UNIQUE constraint
    await database.execute(
        f"""CREATE TABLE {DB_TABLE_PLAYLOG}(
            [id] INTEGER PRIMARY KEY AUTOINCREMENT,
            [item_id] TEXT NOT NULL,
            [provider] TEXT NOT NULL,
            [media_type] TEXT NOT NULL,
            [name] TEXT NOT NULL,
            [image] json,
            [timestamp] INTEGER DEFAULT 0,
            [fully_played] BOOLEAN,
            [seconds_played] INTEGER,
            UNIQUE(item_id, provider, media_type));"""
    )
    # replay the column/index additions of migrations 22, 24 and 41
    await database.execute(f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN userid TEXT")
    await database.execute(
        f"CREATE UNIQUE INDEX {DB_TABLE_PLAYLOG}_unique_idx "
        f"ON {DB_TABLE_PLAYLOG}(item_id,provider,media_type,userid)"
    )
    await database.execute(f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN queue_id TEXT")
    await database.execute(
        f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN user_initiated BOOLEAN NOT NULL DEFAULT 1"
    )
    await database.execute(
        f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN playback_speed REAL NOT NULL DEFAULT 1.0"
    )
    await database.commit()
    return database


async def _create_current_playlog_table(database: DatabaseConnection) -> None:
    """Create the playlog table with the current schema (mirrors __create_database_tables)."""
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_PLAYLOG}(
            [id] INTEGER PRIMARY KEY AUTOINCREMENT,
            [item_id] TEXT NOT NULL,
            [provider] TEXT NOT NULL,
            [media_type] TEXT NOT NULL,
            [name] TEXT NOT NULL,
            [image] json,
            [timestamp] INTEGER DEFAULT 0,
            [fully_played] BOOLEAN,
            [seconds_played] INTEGER,
            [userid] TEXT NOT NULL,
            [queue_id] TEXT,
            [user_initiated] BOOLEAN NOT NULL DEFAULT 1,
            [playback_speed] REAL NOT NULL DEFAULT 1.0,
            UNIQUE(item_id, provider, media_type, userid));"""
    )


def _mass_mock() -> MagicMock:
    """Return a mass mock that supports the (awaited) calls done by migrate_database."""
    mass = MagicMock()
    mass.cache.clear = AsyncMock()
    return mass


async def test_migration_rebuilds_playlog_with_legacy_constraint(tmp_path: Path) -> None:
    """The leftover 3-column UNIQUE constraint is removed by rebuilding the playlog table."""
    database = await _create_legacy_playlog_database(str(tmp_path / "library.db"))
    try:
        # seed entries: two users sharing one item, plus a pre-userid (NULL) legacy row
        await database.insert(
            DB_TABLE_PLAYLOG,
            {
                "item_id": "123",
                "provider": "library",
                "media_type": "track",
                "name": "test track",
                "timestamp": 100,
                "userid": "user1",
            },
        )
        await database.insert(
            DB_TABLE_PLAYLOG,
            {
                "item_id": "legacy",
                "provider": "library",
                "media_type": "track",
                "name": "legacy row without userid",
                "timestamp": 50,
                "userid": None,
            },
        )
        # the legacy constraint blocks logging the same item for a second user
        with pytest.raises(Exception, match="UNIQUE constraint failed"):
            await database.insert(
                DB_TABLE_PLAYLOG,
                {
                    "item_id": "123",
                    "provider": "library",
                    "media_type": "track",
                    "name": "test track",
                    "timestamp": 200,
                    "userid": "user2",
                },
            )

        await migrate_database(
            _mass_mock(),
            database,
            MagicMock(),  # logger
            prev_version=46,
            create_tables=lambda: _create_current_playlog_table(database),
        )

        # existing (attributable) entries survived the rebuild, NULL-userid rows are dropped
        rows = await database.get_rows(DB_TABLE_PLAYLOG)
        assert [(row["item_id"], row["userid"], row["timestamp"]) for row in rows] == [
            ("123", "user1", 100)
        ]
        # the same item can now be logged for a second user...
        await database.insert(
            DB_TABLE_PLAYLOG,
            {
                "item_id": "123",
                "provider": "library",
                "media_type": "track",
                "name": "test track",
                "timestamp": 200,
                "userid": "user2",
            },
        )
        # ...and the per-user upsert used by the playlog writers resolves the conflict
        await database.execute(
            f"INSERT INTO {DB_TABLE_PLAYLOG} "
            "(item_id, provider, media_type, name, userid, timestamp) "
            "VALUES ('123', 'library', 'track', 'test track', 'user2', 300) "
            "ON CONFLICT(item_id, provider, media_type, userid) "
            "DO UPDATE SET timestamp = excluded.timestamp"
        )
        row = await database.get_row(DB_TABLE_PLAYLOG, {"userid": "user2"})
        assert row is not None
        assert row["timestamp"] == 300
    finally:
        await database.close()


async def test_migration_keeps_playlog_with_current_constraint(tmp_path: Path) -> None:
    """A playlog table that already has the 4-column constraint is left untouched."""
    database = DatabaseConnection(str(tmp_path / "library.db"))
    await database.setup()
    try:
        await _create_current_playlog_table(database)
        await database.insert(
            DB_TABLE_PLAYLOG,
            {
                "item_id": "123",
                "provider": "library",
                "media_type": "track",
                "name": "test track",
                "timestamp": 100,
                "userid": "user1",
            },
        )
        create_tables = AsyncMock()

        await migrate_database(
            _mass_mock(),
            database,
            MagicMock(),  # logger
            prev_version=46,
            create_tables=create_tables,
        )

        # no rebuild was triggered and the data is untouched
        create_tables.assert_not_awaited()
        rows = await database.get_rows(DB_TABLE_PLAYLOG)
        assert [(row["item_id"], row["userid"]) for row in rows] == [("123", "user1")]
    finally:
        await database.close()
