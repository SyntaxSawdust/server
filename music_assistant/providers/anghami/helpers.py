"""Authentication and HTTP client helpers for the Anghami music provider."""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse

from aiohttp import ClientError, ClientResponseError
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    RateLimited,
    ResourceTemporarilyUnavailable,
)

from music_assistant.helpers.json import json_loads
from music_assistant.helpers.throttle_retry import ThrottlerManager, throttle_with_retries

from .constants import (
    API_BASE_URL,
    AUTHORIZE_URL,
    OAUTH_SCOPES,
    TOKEN_REFRESH_BUFFER,
    TOKEN_REFRESH_URL,
    TOKEN_URL,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from aiohttp import ClientSession

    from music_assistant.providers.anghami.provider import AnghamiProvider


def generate_code_verifier() -> str:
    """Generate a PKCE code verifier (43-128 URL-safe characters)."""
    return secrets.token_urlsafe(64)


def generate_code_challenge(code_verifier: str) -> str:
    """Derive the S256 PKCE code challenge for the given verifier."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_authorize_url(client_id: str, redirect_uri: str, code_challenge: str, state: str) -> str:
    """Build the Anghami OAuth2 authorization URL for the PKCE flow."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": OAUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def extract_code_from_url(redirected_url: str, expected_state: str) -> str:
    """
    Extract the authorization code from the URL the user was redirected to.

    :param redirected_url: The full URL the user pasted after authenticating.
    :param expected_state: The state value generated when the login was started.
    """
    query = parse_qs(urlparse(redirected_url.strip()).query)
    if error := query.get("error"):
        raise LoginFailed(f"Authentication failed: {error[0]}")
    code = query.get("code", [None])[0]
    state = query.get("state", [None])[0]
    if not code:
        raise LoginFailed("No authorization code found in the provided URL")
    if expected_state and state != expected_state:
        raise LoginFailed("State mismatch on authentication callback")
    return code


async def exchange_code(
    http_session: ClientSession,
    client_id: str,
    code: str,
    code_verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """
    Exchange an authorization code for an Anghami token set.

    :param http_session: The shared aiohttp session.
    :param client_id: The developer application client id.
    :param code: The authorization code returned by Anghami.
    :param code_verifier: The PKCE verifier generated when the login was started.
    :param redirect_uri: The redirect URI registered for the application.
    """
    payload = {
        "code": code,
        "redirectUri": redirect_uri,
        "clientId": client_id,
        "codeVerifier": code_verifier,
    }
    return await _post_token(http_session, TOKEN_URL, payload)


class AnghamiAuthManager:
    """Holds and refreshes the OAuth tokens for an Anghami provider instance."""

    def __init__(
        self,
        http_session: ClientSession,
        client_id: str,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        config_updater: Callable[[dict[str, Any]], None],
    ) -> None:
        """Initialize the auth manager with the persisted token set."""
        self._http_session = http_session
        self._client_id = client_id
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self._config_updater = config_updater

    @property
    def access_token(self) -> str:
        """Return the current access token."""
        return self._access_token

    async def ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing it first if it is about to expire."""
        if time.time() < self._expires_at - TOKEN_REFRESH_BUFFER:
            return self._access_token
        await self._refresh()
        return self._access_token

    async def _refresh(self) -> None:
        """Refresh the access token using the stored refresh token."""
        payload = {"refreshToken": self._refresh_token, "clientId": self._client_id}
        token = await _post_token(self._http_session, TOKEN_REFRESH_URL, payload)
        self._apply_token(token)

    def _apply_token(self, token: dict[str, Any]) -> None:
        """Store a freshly issued token set and persist it to the provider config."""
        self._access_token = token["accessToken"]
        # Anghami may rotate the refresh token; keep the latest one.
        self._refresh_token = token.get("refreshToken", self._refresh_token)
        self._expires_at = time.time() + int(token.get("expiresIn", 3600))
        self._config_updater(
            {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expires_at": self._expires_at,
            }
        )


class AnghamiClient:
    """Thin throttled HTTP client for the Anghami SDK REST API."""

    throttler = ThrottlerManager(rate_limit=5, period=1)

    def __init__(self, provider: AnghamiProvider) -> None:
        """Initialize the client for the given provider instance."""
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a GET request against the Anghami API and return the parsed body."""
        return await self._request("GET", path, params=params)

    async def post(self, path: str, json_body: dict[str, Any]) -> dict[str, Any]:
        """Perform a POST request against the Anghami API and return the parsed body."""
        return await self._request("POST", path, json_body=json_body)

    async def paginate(
        self, path: str, items_key: str, params: dict[str, Any] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Yield every item from a paginated Anghami list endpoint.

        :param path: The API path to request.
        :param items_key: The response key holding the list of items.
        :param params: Optional base query parameters (page_size is added automatically).
        """
        query: dict[str, Any] = dict(params or {})
        query.setdefault("page_size", self.provider.page_size)
        page_token: str | None = None
        while True:
            if page_token:
                query["page_token"] = page_token
            data = await self.get(path, params=query)
            for item in data.get(items_key, []):
                yield item
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    @throttle_with_retries
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single authenticated request with error mapping."""
        access_token = await self.provider.auth.ensure_valid_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            async with self.mass.http_session.request(
                method,
                f"{API_BASE_URL}{path}",
                headers=headers,
                params=_encode_params(params),
                json=json_body,
            ) as response:
                if response.status == 401:
                    raise LoginFailed("Anghami authentication failed")
                if response.status == 404:
                    raise MediaNotFoundError(f"Item not found: {path}")
                if response.status == 429:
                    raise RateLimited(
                        "Anghami rate limit reached",
                        backoff_time=_retry_after(dict(response.headers)),
                    )
                if response.status >= 400:
                    raise ResourceTemporarilyUnavailable(f"Anghami API error: {response.status}")
                result: dict[str, Any] = await response.json(loads=json_loads)
                return result
        except (ClientError, TimeoutError) as err:
            raise ResourceTemporarilyUnavailable(f"Anghami request failed: {err}") from err


async def _post_token(
    http_session: ClientSession, url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST to an Anghami auth endpoint and return the unwrapped TokenInfo token object."""
    try:
        async with http_session.post(url, json=payload) as response:
            response.raise_for_status()
            data: dict[str, Any] = await response.json(loads=json_loads)
    except ClientResponseError as err:
        raise LoginFailed(f"Anghami token request failed: {err.status}") from err
    except (ClientError, TimeoutError) as err:
        raise LoginFailed(f"Anghami token request failed: {err}") from err
    token = data.get("token")
    if not token or "accessToken" not in token:
        raise LoginFailed("Anghami token response did not contain a token")
    return token


def _encode_params(params: dict[str, Any] | None) -> list[tuple[str, str]] | None:
    """Encode query params, expanding list values into repeated key/value pairs."""
    if not params:
        return None
    encoded: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            encoded.extend((key, str(item)) for item in value)
        else:
            encoded.append((key, str(value)))
    return encoded


def _retry_after(headers: dict[str, str]) -> int:
    """Derive a backoff time in seconds from rate-limit response headers."""
    if reset := headers.get("X-RateLimit-Reset"):
        try:
            return max(0, int(reset) - int(time.time()))
        except ValueError:
            return 30
    return 30
