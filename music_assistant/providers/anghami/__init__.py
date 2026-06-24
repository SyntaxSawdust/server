"""Anghami music provider for Music Assistant."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType
from music_assistant_models.errors import LoginFailed

from music_assistant.constants import CONF_ENTRY_UNOFFICIAL_PROVIDER
from music_assistant.helpers.auth import AuthenticationHelper

from .constants import (
    CONF_ACCESS_TOKEN,
    CONF_ACTION_CLEAR_AUTH,
    CONF_ACTION_COMPLETE_LOGIN,
    CONF_ACTION_START_LOGIN,
    CONF_CLIENT_ID,
    CONF_CODE_VERIFIER,
    CONF_EXPIRY_TIME,
    CONF_REDIRECT_URI,
    CONF_REDIRECTED_URL,
    CONF_REFRESH_TOKEN,
    CONF_STATE,
    LABEL_COMPLETE_LOGIN,
    LABEL_OK,
    LABEL_REDIRECTED_URL,
    LABEL_START_LOGIN,
)
from .helpers import (
    build_authorize_url,
    exchange_code,
    extract_code_from_url,
    generate_code_challenge,
    generate_code_verifier,
)
from .provider import AnghamiProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    if not config.get_value(CONF_ACCESS_TOKEN):
        raise LoginFailed("Anghami provider is not authenticated")
    return AnghamiProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    Return config entries to setup this provider.

    :param mass: The MusicAssistant instance.
    :param instance_id: Id of an existing provider instance (None if new instance setup).
    :param action: Optional action key called from the config entries UI.
    :param values: The (intermediate) raw values for config entries sent with the action.
    """
    assert values is not None

    if action == CONF_ACTION_START_LOGIN:
        await _handle_start_login(mass, values)
    elif action == CONF_ACTION_COMPLETE_LOGIN:
        await _handle_complete_login(mass, values)
    elif action == CONF_ACTION_CLEAR_AUTH:
        for key in (CONF_ACCESS_TOKEN, CONF_REFRESH_TOKEN, CONF_EXPIRY_TIME):
            values[key] = None

    credential_entries = (
        ConfigEntry(
            key=CONF_CLIENT_ID,
            type=ConfigEntryType.STRING,
            required=True,
            value=values.get(CONF_CLIENT_ID),
        ),
        ConfigEntry(
            key=CONF_REDIRECT_URI,
            type=ConfigEntryType.STRING,
            required=True,
            value=values.get(CONF_REDIRECT_URI),
        ),
    )

    if values.get(CONF_ACCESS_TOKEN):
        auth_entries: tuple[ConfigEntry, ...] = (
            ConfigEntry(key=LABEL_OK, type=ConfigEntryType.LABEL),
            ConfigEntry(
                key=CONF_ACTION_CLEAR_AUTH,
                type=ConfigEntryType.ACTION,
                action=CONF_ACTION_CLEAR_AUTH,
                value=None,
            ),
        )
    else:
        login_started = action == CONF_ACTION_START_LOGIN
        auth_entries = (
            ConfigEntry(
                key=LABEL_START_LOGIN,
                type=ConfigEntryType.LABEL,
                hidden=login_started,
            ),
            ConfigEntry(
                key=CONF_ACTION_START_LOGIN,
                type=ConfigEntryType.ACTION,
                action=CONF_ACTION_START_LOGIN,
                depends_on=CONF_REDIRECT_URI,
                hidden=login_started,
            ),
            ConfigEntry(
                key=LABEL_REDIRECTED_URL,
                type=ConfigEntryType.LABEL,
                hidden=not login_started,
            ),
            ConfigEntry(
                key=CONF_REDIRECTED_URL,
                type=ConfigEntryType.STRING,
                depends_on=CONF_ACTION_START_LOGIN,
                value=values.get(CONF_REDIRECTED_URL),
                hidden=not login_started,
            ),
            ConfigEntry(
                key=LABEL_COMPLETE_LOGIN,
                type=ConfigEntryType.LABEL,
                hidden=not login_started,
            ),
            ConfigEntry(
                key=CONF_ACTION_COMPLETE_LOGIN,
                type=ConfigEntryType.ACTION,
                action=CONF_ACTION_COMPLETE_LOGIN,
                depends_on=CONF_REDIRECTED_URL,
                hidden=not login_started,
            ),
            ConfigEntry(
                key=CONF_CODE_VERIFIER,
                type=ConfigEntryType.SECURE_STRING,
                hidden=True,
                value=values.get(CONF_CODE_VERIFIER),
            ),
            ConfigEntry(
                key=CONF_STATE,
                type=ConfigEntryType.STRING,
                hidden=True,
                value=values.get(CONF_STATE),
            ),
        )

    return (
        CONF_ENTRY_UNOFFICIAL_PROVIDER,
        *credential_entries,
        *auth_entries,
        ConfigEntry(
            key=CONF_ACCESS_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            hidden=True,
            value=values.get(CONF_ACCESS_TOKEN),
        ),
        ConfigEntry(
            key=CONF_REFRESH_TOKEN,
            type=ConfigEntryType.SECURE_STRING,
            hidden=True,
            value=values.get(CONF_REFRESH_TOKEN),
        ),
        ConfigEntry(
            key=CONF_EXPIRY_TIME,
            type=ConfigEntryType.STRING,
            hidden=True,
            value=values.get(CONF_EXPIRY_TIME),
        ),
    )


async def _handle_start_login(mass: MusicAssistant, values: dict[str, ConfigValueType]) -> None:
    """Generate the PKCE challenge and redirect the user to the Anghami login page."""
    client_id = values.get(CONF_CLIENT_ID)
    redirect_uri = values.get(CONF_REDIRECT_URI)
    if not client_id or not redirect_uri:
        raise LoginFailed("Please provide the Client ID and Redirect URI first")
    code_verifier = generate_code_verifier()
    state = generate_code_verifier()
    values[CONF_CODE_VERIFIER] = code_verifier
    values[CONF_STATE] = state
    auth_url = build_authorize_url(
        str(client_id), str(redirect_uri), generate_code_challenge(code_verifier), state
    )
    async with AuthenticationHelper(mass, str(values["session_id"])) as auth_helper:
        # Anghami redirects to the registered URI; the user copies that URL back into the next field.
        auth_helper.send_url(auth_url)
        await asyncio.sleep(15)


async def _handle_complete_login(mass: MusicAssistant, values: dict[str, ConfigValueType]) -> None:
    """Exchange the pasted redirect URL for an Anghami token set."""
    code = extract_code_from_url(str(values.get(CONF_REDIRECTED_URL)), str(values.get(CONF_STATE)))
    token = await exchange_code(
        mass.http_session,
        str(values.get(CONF_CLIENT_ID)),
        code,
        str(values.get(CONF_CODE_VERIFIER)),
        str(values.get(CONF_REDIRECT_URI)),
    )
    values[CONF_ACCESS_TOKEN] = token["accessToken"]
    values[CONF_REFRESH_TOKEN] = token.get("refreshToken")
    values[CONF_EXPIRY_TIME] = str(time.time() + int(token.get("expiresIn", 3600)))
    values[CONF_CODE_VERIFIER] = None
    values[CONF_STATE] = None
