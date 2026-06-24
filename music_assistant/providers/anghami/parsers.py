"""Parsers that convert Anghami API objects into Music Assistant media items."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import AlbumType, ContentType, ExternalID, ImageType, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    ItemMapping,
    MediaItemImage,
    Playlist,
    ProviderMapping,
    Track,
    UniqueList,
)

if TYPE_CHECKING:
    from music_assistant_models.media_items import MediaItemType

    from .provider import AnghamiProvider

ALBUM_TYPE_MAP: dict[str, AlbumType] = {
    "ALBUM_TYPE_ALBUM": AlbumType.ALBUM,
    "ALBUM_TYPE_SINGLE": AlbumType.SINGLE,
    "ALBUM_TYPE_COMPILATION": AlbumType.COMPILATION,
    "ALBUM_TYPE_EP": AlbumType.EP,
}


def parse_artist(provider: AnghamiProvider, artist: dict[str, Any]) -> Artist:
    """Build a full Artist from an Anghami catalog Artist object."""
    artist_id = _id_value(artist.get("id"))
    item = Artist(
        item_id=artist_id,
        provider=provider.instance_id,
        name=_localized(artist.get("name")),
        provider_mappings={_provider_mapping(provider, artist_id)},
    )
    if bio := _localized(artist.get("bio")):
        item.metadata.description = bio
    if genres := artist.get("genres"):
        item.metadata.genres = set(genres)
    if images := _images(provider, artist.get("artwork")):
        item.metadata.images = images
    return item


def parse_album(provider: AnghamiProvider, album: dict[str, Any]) -> Album:
    """Build a full Album from an Anghami catalog Album object."""
    album_id = _id_value(album.get("id"))
    item = Album(
        item_id=album_id,
        provider=provider.instance_id,
        name=_localized(album.get("title")),
        provider_mappings={_provider_mapping(provider, album_id)},
        artists=UniqueList(_artist_mappings(provider, album.get("artists"))),
    )
    item.album_type = ALBUM_TYPE_MAP.get(album.get("albumType", ""), AlbumType.UNKNOWN)
    if year := _release_year(album.get("releaseDate")):
        item.year = year
    if release_date := _release_date(album.get("releaseDate")):
        item.metadata.release_date = release_date
    if upc := album.get("upc"):
        item.external_ids.add((ExternalID.BARCODE, upc))
    if (popularity := album.get("popularity")) is not None:
        item.metadata.popularity = popularity
    if genres := album.get("genres"):
        item.metadata.genres = set(genres)
    if images := _images(provider, album.get("artwork")):
        item.metadata.images = images
    return item


def parse_track(
    provider: AnghamiProvider, song: dict[str, Any], album_mapping: ItemMapping | None = None
) -> Track:
    """
    Build a full Track from an Anghami catalog Song object.

    :param provider: The provider instance.
    :param song: The Anghami Song object.
    :param album_mapping: Optional album mapping to attach (Song carries only an album id).
    """
    track_id = _id_value(song.get("id"))
    item = Track(
        item_id=track_id,
        provider=provider.instance_id,
        name=_localized(song.get("title")),
        duration=int(song.get("durationMs", 0)) // 1000,
        provider_mappings={_provider_mapping(provider, track_id)},
        artists=UniqueList(_artist_mappings(provider, song.get("artists"))),
        disc_number=int(song.get("discNumber", 0) or 0),
        track_number=int(song.get("trackNumber", 0) or 0),
    )
    if isrc := song.get("isrc"):
        item.external_ids.add((ExternalID.ISRC, isrc))
    if album_mapping is not None:
        item.album = album_mapping
    item.metadata.explicit = bool(song.get("isExplicit", False))
    if (popularity := song.get("popularity")) is not None:
        item.metadata.popularity = popularity
    if images := _images(provider, song.get("artwork")):
        item.metadata.images = images
    return item


def parse_playlist(provider: AnghamiProvider, playlist: dict[str, Any]) -> Playlist:
    """Build a full Playlist from an Anghami PlaylistService Playlist object."""
    playlist_id = _id_value(playlist.get("id"))
    item = Playlist(
        item_id=playlist_id,
        provider=provider.instance_id,
        name=_localized(playlist.get("name")),
        owner=playlist.get("ownerName", ""),
        provider_mappings={_provider_mapping(provider, playlist_id)},
    )
    # Anghami exposes no playlist mutation endpoints, so playlists are never editable.
    item.is_editable = False
    if description := _localized(playlist.get("description")):
        item.metadata.description = description
    if images := _single_image(provider, playlist.get("coverArtwork")):
        item.metadata.images = images
    return item


def parse_content(provider: AnghamiProvider, content: dict[str, Any]) -> MediaItemType | None:
    """
    Build a lightweight media item from an Anghami Content wrapper.

    Content wrappers only carry an id, title, subtitle and a single artwork; full details are
    fetched on demand via the typed get_* endpoints. Returns None for unsupported content types.
    """
    title = content.get("title", "")
    images = _single_image(provider, content.get("artwork"))
    if song_id := _id_value(content.get("songId")):
        track = Track(
            item_id=song_id,
            provider=provider.instance_id,
            name=title,
            provider_mappings={_provider_mapping(provider, song_id)},
        )
        if images:
            track.metadata.images = images
        return track
    if album_id := _id_value(content.get("albumId")):
        album = Album(
            item_id=album_id,
            provider=provider.instance_id,
            name=title,
            provider_mappings={_provider_mapping(provider, album_id)},
        )
        if images:
            album.metadata.images = images
        return album
    if artist_id := _id_value(content.get("artistId")):
        artist = Artist(
            item_id=artist_id,
            provider=provider.instance_id,
            name=title,
            provider_mappings={_provider_mapping(provider, artist_id)},
        )
        if images:
            artist.metadata.images = images
        return artist
    if playlist_id := _id_value(content.get("playlistId")):
        playlist = Playlist(
            item_id=playlist_id,
            provider=provider.instance_id,
            name=title,
            owner=content.get("subtitle", ""),
            provider_mappings={_provider_mapping(provider, playlist_id)},
        )
        playlist.is_editable = False
        if images:
            playlist.metadata.images = images
        return playlist
    return None


def _provider_mapping(provider: AnghamiProvider, item_id: str) -> ProviderMapping:
    """Build the standard ProviderMapping for an Anghami item."""
    return ProviderMapping(
        item_id=item_id,
        provider_domain=provider.domain,
        provider_instance=provider.instance_id,
        available=True,
        audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
    )


def _artist_mappings(
    provider: AnghamiProvider, artists: list[dict[str, Any]] | None
) -> list[ItemMapping]:
    """Build artist ItemMappings from an Anghami artist reference list."""
    mappings: list[ItemMapping] = []
    for artist in artists or []:
        if artist_id := _id_value(artist.get("id")):
            mappings.append(
                ItemMapping(
                    media_type=MediaType.ARTIST,
                    item_id=artist_id,
                    provider=provider.instance_id,
                    name=_localized(artist.get("name")),
                )
            )
    return mappings


def _images(
    provider: AnghamiProvider, artwork: list[dict[str, Any]] | None
) -> UniqueList[MediaItemImage]:
    """Build a thumbnail image list from an Anghami artwork array."""
    images: UniqueList[MediaItemImage] = UniqueList()
    for image in artwork or []:
        if url := image.get("url"):
            images.append(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=url,
                    provider=provider.instance_id,
                    remotely_accessible=True,
                )
            )
            break
    return images


def _single_image(
    provider: AnghamiProvider, image: dict[str, Any] | None
) -> UniqueList[MediaItemImage]:
    """Build a thumbnail image list from a single Anghami Image object."""
    if image and (url := image.get("url")):
        return UniqueList(
            [
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=url,
                    provider=provider.instance_id,
                    remotely_accessible=True,
                )
            ]
        )
    return UniqueList()


def _id_value(wrapper: dict[str, Any] | None) -> str:
    """Return the string value of an Anghami typed-id wrapper, or an empty string."""
    if wrapper and (value := wrapper.get("value")):
        return str(value)
    return ""


def _localized(wrapper: dict[str, Any] | None) -> str:
    """Return the display string of an Anghami LocalizedString wrapper."""
    if not wrapper:
        return ""
    return str(wrapper.get("value") or wrapper.get("originalValue") or "")


def _release_year(release_date: str | None) -> int | None:
    """Extract the release year from a YYYY-MM-DD string."""
    if release_date and len(release_date) >= 4 and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def _release_date(release_date: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD release date string into a datetime."""
    if not release_date:
        return None
    try:
        return datetime.fromisoformat(release_date)
    except ValueError:
        return None
