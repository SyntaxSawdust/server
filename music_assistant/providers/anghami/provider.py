"""Anghami music provider implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import LoginFailed, UnplayableMediaError
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    Playlist,
    RecommendationFolder,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.controllers.cache import use_cache
from music_assistant.models.music_provider import MusicProvider

from .constants import (
    CACHE_CATEGORY_RECOMMENDATIONS,
    CONF_ACCESS_TOKEN,
    CONF_CLIENT_ID,
    CONF_EXPIRY_TIME,
    CONF_REFRESH_TOKEN,
    CONTENT_TYPE_ALBUM,
    CONTENT_TYPE_ARTIST,
    CONTENT_TYPE_PLAYLIST,
    CONTENT_TYPE_SONG,
    DEFAULT_PAGE_SIZE,
    DRM_SCHEME_CLEAR,
    MAX_PAGE_SIZE,
    SUPPORTED_FEATURES,
)
from .helpers import AnghamiAuthManager, AnghamiClient
from .parsers import parse_album, parse_artist, parse_content, parse_playlist, parse_track

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import MediaItemType
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant

SEARCH_TYPE_MAP: dict[MediaType, str] = {
    MediaType.ARTIST: CONTENT_TYPE_ARTIST,
    MediaType.ALBUM: CONTENT_TYPE_ALBUM,
    MediaType.TRACK: CONTENT_TYPE_SONG,
    MediaType.PLAYLIST: CONTENT_TYPE_PLAYLIST,
}


class AnghamiProvider(MusicProvider):
    """Streaming music provider for Anghami."""

    auth: AnghamiAuthManager

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize the Anghami provider instance."""
        super().__init__(mass, manifest, config, SUPPORTED_FEATURES)
        self.client = AnghamiClient(self)
        self.page_size = DEFAULT_PAGE_SIZE

    async def handle_async_init(self) -> None:
        """Set up the auth manager from the persisted token set."""
        access_token = self.config.get_value(CONF_ACCESS_TOKEN)
        refresh_token = self.config.get_value(CONF_REFRESH_TOKEN)
        client_id = self.config.get_value(CONF_CLIENT_ID)
        if not access_token or not refresh_token or not client_id:
            raise LoginFailed("Anghami provider is not authenticated")
        expiry = self.config.get_value(CONF_EXPIRY_TIME)
        self.auth = AnghamiAuthManager(
            http_session=self.mass.http_session,
            client_id=str(client_id),
            access_token=str(access_token),
            refresh_token=str(refresh_token),
            expires_at=float(expiry) if expiry else 0.0,
            config_updater=self._update_auth_config,
        )
        # Refresh up front so a stale stored token is renewed before the first request.
        await self.auth.ensure_valid_token()

    @use_cache(3600 * 24)
    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """
        Perform a search on Anghami.

        :param search_query: The search query.
        :param media_types: The media types to include.
        :param limit: The maximum number of items to return per media type.
        """
        types = [SEARCH_TYPE_MAP[mt] for mt in media_types if mt in SEARCH_TYPE_MAP]
        if not types:
            return SearchResults()
        params = {
            "q": search_query,
            "types": types,
            "page_size": min(MAX_PAGE_SIZE, max(limit * len(types), limit)),
        }
        data = await self.client.get("/v1/discovery/search", params)
        artists: list[Artist] = []
        albums: list[Album] = []
        tracks: list[Track] = []
        playlists: list[Playlist] = []
        for content in data.get("results", []):
            item = parse_content(self, content)
            if isinstance(item, Artist) and len(artists) < limit:
                artists.append(item)
            elif isinstance(item, Album) and len(albums) < limit:
                albums.append(item)
            elif isinstance(item, Track) and len(tracks) < limit:
                tracks.append(item)
            elif isinstance(item, Playlist) and len(playlists) < limit:
                playlists.append(item)
        return SearchResults(artists=artists, albums=albums, tracks=tracks, playlists=playlists)

    # Catalog metadata is effectively immutable, so cache it for a long time (6 months).
    @use_cache(3600 * 24 * 180)
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get full artist details by id."""
        data = await self.client.get(f"/v1/music/artists/{prov_artist_id}")
        return parse_artist(self, data["artist"])

    @use_cache(3600 * 24 * 180)
    async def get_album(self, prov_album_id: str) -> Album:
        """Get full album details by id."""
        data = await self.client.get(f"/v1/music/albums/{prov_album_id}")
        return parse_album(self, data["album"])

    @use_cache(3600 * 24 * 180)
    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        data = await self.client.get(f"/v1/music/songs/{prov_track_id}")
        return parse_track(self, data["song"])

    # Playlists (name, cover, track count) change, so keep this shorter.
    @use_cache(3600 * 24)
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full playlist details by id."""
        data = await self.client.get(f"/v1/playlists/{prov_playlist_id}", {"page_size": 1})
        return parse_playlist(self, data["playlist"])

    @use_cache(3600 * 24 * 180, allow_expired_cache=True)
    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Get all tracks for the given album id."""
        album = await self.get_album(prov_album_id)
        album_mapping = ItemMapping(
            media_type=MediaType.ALBUM,
            item_id=prov_album_id,
            provider=self.instance_id,
            name=album.name,
        )
        tracks: list[Track] = []
        async for song in self.client.paginate(
            f"/v1/music/albums/{prov_album_id}/tracks", "tracks"
        ):
            tracks.append(parse_track(self, song, album_mapping))
        return tracks

    # Playlists are user-editable (in the Anghami app), so keep the track list fresh.
    @use_cache(60 * 15, allow_expired_cache=True)
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> Sequence[Track]:
        """Get all tracks for the given playlist id."""
        # Anghami paginates by opaque token; the full list is gathered on the first page.
        if page > 0:
            return []
        items: list[dict[str, Any]] = []
        song_ids: list[str] = []
        async for item in self.client.paginate(f"/v1/playlists/{prov_playlist_id}", "items"):
            content = item.get("content", {})
            song_id = content.get("songId", {}).get("value")
            if not song_id:
                continue
            items.append(item)
            song_ids.append(song_id)
        songs = await self._batch_get_songs(song_ids)
        tracks: list[Track] = []
        for index, item in enumerate(items):
            song = songs.get(item["content"]["songId"]["value"])
            if song is None:
                continue
            track = parse_track(self, song)
            track.position = int(item.get("position", index + 1))
            tracks.append(track)
        return tracks

    # Refresh daily so newly released albums appear within a day.
    @use_cache(3600 * 24, allow_expired_cache=True)
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Get all albums for the given artist."""
        albums: list[Album] = []
        async for album in self.client.paginate(
            f"/v1/music/artists/{prov_artist_id}/discography", "albums"
        ):
            albums.append(parse_album(self, album))
        return albums

    @use_cache(3600 * 24 * 7, allow_expired_cache=True)
    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """Get the most popular tracks for the given artist."""
        tracks: list[Track] = []
        async for song in self.client.paginate(
            f"/v1/music/artists/{prov_artist_id}/top-tracks", "tracks"
        ):
            tracks.append(parse_track(self, song))
        return tracks

    @use_cache(3600 * 24, allow_expired_cache=True)
    async def get_similar_artists(self, prov_artist_id: str, limit: int = 25) -> list[Artist]:
        """Get a list of artists related to the given artist."""
        artists: list[Artist] = []
        async for artist in self.client.paginate(
            f"/v1/music/artists/{prov_artist_id}/related", "artists"
        ):
            artists.append(parse_artist(self, artist))
            if len(artists) >= limit:
                break
        return artists

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Retrieve the followed artists from the user's library."""
        async for content in self.client.paginate("/v1/library/following/artists", "items"):
            item = parse_content(self, content)
            if isinstance(item, Artist):
                yield item

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Retrieve the liked albums from the user's library."""
        ids = [
            album_id
            async for content in self.client.paginate(
                "/v1/library/likes", "items", {"content_type": CONTENT_TYPE_ALBUM}
            )
            if (album_id := content.get("albumId", {}).get("value"))
        ]
        for chunk in _chunked(ids, 100):
            albums = await self._batch_get_albums(chunk)
            for album_id in chunk:
                if album := albums.get(album_id):
                    yield parse_album(self, album)

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Retrieve the liked tracks from the user's library."""
        ids = [
            song_id
            async for content in self.client.paginate(
                "/v1/library/likes", "items", {"content_type": CONTENT_TYPE_SONG}
            )
            if (song_id := content.get("songId", {}).get("value"))
        ]
        for chunk in _chunked(ids, 100):
            songs = await self._batch_get_songs(chunk)
            for song_id in chunk:
                if song := songs.get(song_id):
                    yield parse_track(self, song)

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Retrieve the user's playlists."""
        async for playlist in self.client.paginate("/v1/playlists", "playlists"):
            yield parse_playlist(self, playlist)

    @use_cache(3600, category=CACHE_CATEGORY_RECOMMENDATIONS)
    async def recommendations(self) -> list[RecommendationFolder]:
        """Get editorial recommendations (featured, new releases and charts)."""
        folders: list[RecommendationFolder] = []
        folders.extend(await self._featured_folders())
        folders.extend(await self._new_release_folders())
        folders.extend(await self._chart_folders())
        return folders

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get the stream details for a track."""
        async with self.client.throttler.bypass():
            data = await self.client.post(
                "/v1/streaming/music",
                {"songId": {"value": item_id}, "maxQuality": "AUDIO_QUALITY_LOSSLESS"},
            )
        stream = data.get("stream", {})
        drm_scheme = stream.get("drmScheme") or DRM_SCHEME_CLEAR
        if drm_scheme != DRM_SCHEME_CLEAR:
            self.logger.debug("Track %s is protected with %s DRM", item_id, drm_scheme)
            raise UnplayableMediaError(
                f"Anghami track {item_id} is DRM-protected ({drm_scheme}) and cannot be played"
            )
        manifest_url = stream.get("manifestUrl")
        if not manifest_url:
            raise UnplayableMediaError(f"No stream URL returned for Anghami track {item_id}")
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
            media_type=MediaType.TRACK,
            stream_type=StreamType.HLS if "m3u8" in manifest_url else StreamType.HTTP,
            path=manifest_url,
            allow_seek=True,
            can_seek=True,
            expiration=int(stream.get("expiresInSeconds", 0)),
        )

    def _update_auth_config(self, auth_info: dict[str, Any]) -> None:
        """Persist a refreshed token set back into the provider config."""
        self._update_config_value(CONF_ACCESS_TOKEN, auth_info["access_token"], encrypted=True)
        self._update_config_value(CONF_REFRESH_TOKEN, auth_info["refresh_token"], encrypted=True)
        self._update_config_value(CONF_EXPIRY_TIME, str(auth_info["expires_at"]))

    async def _batch_get_songs(self, song_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate full Song objects for the given ids via the batch endpoint."""
        return await self._batch_get(song_ids, "/v1/music/songs:batchGet", "songIdList", "song")

    async def _batch_get_albums(self, album_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate full Album objects for the given ids via the batch endpoint."""
        return await self._batch_get(album_ids, "/v1/music/albums:batchGet", "albumIdList", "album")

    async def _batch_get(
        self, ids: list[str], path: str, id_list_key: str, object_key: str
    ) -> dict[str, dict[str, Any]]:
        """Resolve a set of ids to full objects, keyed by id, skipping per-item errors."""
        result: dict[str, dict[str, Any]] = {}
        for chunk in _chunked(ids, 100):
            body = {id_list_key: [{"value": item_id} for item_id in chunk]}
            data = await self.client.post(path, body)
            for item_id, entry in data.get("results", {}).items():
                if object_key in entry:
                    result[item_id] = entry[object_key]
        return result

    async def _featured_folders(self) -> list[RecommendationFolder]:
        """Build recommendation folders from the featured sections."""
        data = await self.client.get("/v1/discovery/browse/featured")
        folders: list[RecommendationFolder] = []
        for section in data.get("sections", []):
            section_id = section.get("id", {}).get("value", "")
            folders.append(
                self._build_folder(
                    item_id=f"featured_{section_id}",
                    name=section.get("title", "Featured"),
                    subtitle=section.get("subtitle", ""),
                    contents=section.get("items", []),
                    icon="mdi-star",
                )
            )
        return folders

    async def _new_release_folders(self) -> list[RecommendationFolder]:
        """Build a recommendation folder from the new releases."""
        data = await self.client.get("/v1/discovery/browse/new-releases")
        return [
            self._build_folder(
                item_id="new_releases",
                name="New Releases",
                subtitle="",
                contents=data.get("items", []),
                icon="mdi-new-box",
            )
        ]

    async def _chart_folders(self) -> list[RecommendationFolder]:
        """Build recommendation folders from the charts."""
        data = await self.client.get("/v1/discovery/browse/charts")
        folders: list[RecommendationFolder] = []
        for chart_entry in data.get("charts", []):
            chart = chart_entry.get("chart", {})
            contents = [entry.get("content", {}) for entry in chart_entry.get("entries", [])]
            folders.append(
                self._build_folder(
                    item_id=f"chart_{chart.get('id', {}).get('value', '')}",
                    name=chart.get("name", "Chart"),
                    subtitle=chart.get("description", ""),
                    contents=contents,
                    icon="mdi-chart-line",
                )
            )
        return folders

    def _build_folder(
        self,
        item_id: str,
        name: str,
        subtitle: str,
        contents: Iterable[dict[str, Any]],
        icon: str,
    ) -> RecommendationFolder:
        """Build a RecommendationFolder from a list of Anghami Content wrappers."""
        items: UniqueList[MediaItemType | ItemMapping | BrowseFolder] = UniqueList()
        for content in contents:
            if item := parse_content(self, content):
                items.append(item)
        return RecommendationFolder(
            item_id=item_id,
            name=name,
            provider=self.instance_id,
            items=items,
            subtitle=subtitle,
            icon=icon,
        )


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    """Yield successive chunks of at most ``size`` items."""
    for index in range(0, len(items), size):
        yield items[index : index + size]
