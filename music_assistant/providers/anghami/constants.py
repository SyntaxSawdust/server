"""Constants for the Anghami music provider."""

from __future__ import annotations

from typing import Final

from music_assistant_models.enums import ProviderFeature

DOMAIN: Final[str] = "anghami"

# Single host for all Anghami SDK calls (auth, catalog, discovery, library, streaming).
API_BASE_URL: Final[str] = "https://sdk.anghami.com"
AUTHORIZE_URL: Final[str] = f"{API_BASE_URL}/v1/auth/authorize"
TOKEN_URL: Final[str] = f"{API_BASE_URL}/v1/auth/token"
TOKEN_REFRESH_URL: Final[str] = f"{API_BASE_URL}/v1/auth/token/refresh"

# `read` covers catalog/search/browse/library, `stream` covers (billable) stream acquisition.
OAUTH_SCOPES: Final[str] = "read stream"

# Refresh the access token this many seconds before it actually expires.
TOKEN_REFRESH_BUFFER: Final[int] = 60 * 5

# Pagination (Anghami allows page_size 1-100, default 20).
DEFAULT_PAGE_SIZE: Final[int] = 50
MAX_PAGE_SIZE: Final[int] = 100

# Config entry keys.
CONF_CLIENT_ID: Final[str] = "client_id"
CONF_REDIRECT_URI: Final[str] = "redirect_uri"
CONF_REDIRECTED_URL: Final[str] = "redirected_url"
CONF_ACCESS_TOKEN: Final[str] = "access_token"
CONF_REFRESH_TOKEN: Final[str] = "refresh_token"
CONF_EXPIRY_TIME: Final[str] = "expiry_time"
CONF_CODE_VERIFIER: Final[str] = "code_verifier"
CONF_STATE: Final[str] = "state"

# Config flow action keys.
CONF_ACTION_START_LOGIN: Final[str] = "start_login"
CONF_ACTION_COMPLETE_LOGIN: Final[str] = "complete_login"
CONF_ACTION_CLEAR_AUTH: Final[str] = "clear_auth"

# Labels used in the config flow.
LABEL_START_LOGIN: Final[str] = "start_login_label"
LABEL_REDIRECTED_URL: Final[str] = "redirected_url_label"
LABEL_COMPLETE_LOGIN: Final[str] = "complete_login_label"
LABEL_OK: Final[str] = "label_ok"

# Cache categories.
CACHE_CATEGORY_DEFAULT: Final[int] = 0
CACHE_CATEGORY_RECOMMENDATIONS: Final[int] = 1

# Anghami DRM scheme values; only UNSPECIFIED (clear) is playable without a DRM module.
DRM_SCHEME_CLEAR: Final[str] = "DRM_SCHEME_UNSPECIFIED"

# Map Music Assistant media types to Anghami CONTENT_TYPE_* enum values.
CONTENT_TYPE_SONG: Final[str] = "CONTENT_TYPE_SONG"
CONTENT_TYPE_ALBUM: Final[str] = "CONTENT_TYPE_ALBUM"
CONTENT_TYPE_ARTIST: Final[str] = "CONTENT_TYPE_ARTIST"
CONTENT_TYPE_PLAYLIST: Final[str] = "CONTENT_TYPE_PLAYLIST"

SUPPORTED_FEATURES: Final[set[ProviderFeature]] = {
    ProviderFeature.SEARCH,
    ProviderFeature.BROWSE,
    ProviderFeature.RECOMMENDATIONS,
    ProviderFeature.LIBRARY_ARTISTS,
    ProviderFeature.LIBRARY_ALBUMS,
    ProviderFeature.LIBRARY_TRACKS,
    ProviderFeature.LIBRARY_PLAYLISTS,
    ProviderFeature.ARTIST_ALBUMS,
    ProviderFeature.ARTIST_TOPTRACKS,
    ProviderFeature.SIMILAR_ARTISTS,
}
