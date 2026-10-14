"""Abstract base class for all database implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from database.models import (
    DiscordCredentialRow,
    GmailReplyStateRow,
    GmailSyncStateRow,
    GoogleCredentialRow,
    SlackCredentialRow,
    TelegramCredentialRow,
    User,
)


class BaseDatabase(ABC):
    """Defines the contract for database access layers.

    Any concrete implementation (PostgreSQL, SQLite, etc.) must
    implement all methods below.
    """

    @abstractmethod
    def create_database(self) -> None:
        """Create the application database if it does not exist."""

    @abstractmethod
    def create_tables(self) -> None:
        """Create all application tables if they do not already exist."""

    @abstractmethod
    def find_user_by_id(self, user_id: int) -> Optional[User]:
        """Return the user row for `user_id`, or None if not found."""

    @abstractmethod
    def user_exists(self, email: str) -> bool:
        """Return True if a user with the given email exists."""

    @abstractmethod
    def find_user_by_email(self, email: str) -> Optional[User]:
        """Return the user row for `email`, or None if not found."""

    @abstractmethod
    def create_user(self, *, name: str, email: str, password_hash: str) -> User:
        """Insert a new user and return the created row."""

    @abstractmethod
    def update_user_profile(self, user_id: int, *, name: str) -> User:
        """Update mutable profile fields for a user."""

    @abstractmethod
    def find_google_credentials(self, user_id: int) -> Optional[GoogleCredentialRow]:
        """Return the Google credentials row for a user, or None."""

    @abstractmethod
    def find_user_by_google_id(self, google_id: str) -> Optional[GoogleCredentialRow]:
        """Return the Google credentials row for `google_id`, or None."""

    @abstractmethod
    def save_google_credentials(
        self,
        user_id: int,
        *,
        google_id: str,
        access_token: str,
        refresh_token: Optional[str],
        token_expiry: Optional[datetime],
    ) -> GoogleCredentialRow:
        """Insert or update Google credentials for a user."""

    # ------------------------------------------------------------------
    # Slack Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_slack_credentials(self, team_id: str) -> Optional[SlackCredentialRow]:
        """Return the Slack credentials row for a team, or None."""

    @abstractmethod
    def save_slack_credentials(
        self,
        user_id: int,
        team_id: str,
        *,
        access_token: str,
        refresh_token: Optional[str] = None,
        team_name: Optional[str] = None,
        bot_user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SlackCredentialRow:
        """Insert or update Slack credentials for a workspace."""

    @abstractmethod
    def delete_slack_credentials(self, team_id: str) -> bool:
        """Delete Slack credentials for a workspace. Returns True if deleted."""

    # ------------------------------------------------------------------
    # Discord Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_discord_credentials(
        self, discord_user_id: str, guild_id: str
    ) -> Optional[DiscordCredentialRow]:
        """Return the Discord credentials for a user+guild, or None."""

    @abstractmethod
    def find_discord_credentials_by_guild(self, guild_id: str) -> Optional[DiscordCredentialRow]:
        """Return the first Discord credentials for a guild, or None."""

    @abstractmethod
    def save_discord_credentials(
        self,
        *,
        user_id: int,
        discord_user_id: str,
        guild_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_type: str = "Bearer",
        scopes: str = "identify bot",
        expires_at: Optional[datetime] = None,
        granted_permissions: Optional[str] = None,
    ) -> DiscordCredentialRow:
        """Insert or update Discord credentials for a user+guild."""

    @abstractmethod
    def delete_discord_credentials(self, discord_user_id: str, guild_id: str) -> bool:
        """Delete Discord credentials. Returns True if deleted."""

    @abstractmethod
    def list_discord_guild_ids(self) -> List[str]:
        """Return all distinct connected guild IDs."""

    @abstractmethod
    def discord_guild_connection_exists(self, guild_id: str) -> bool:
        """Return True if a connection exists for the guild."""

    @abstractmethod
    def discord_guild_owned_by_user(self, user_id: int, guild_id: str) -> bool:
        """Return True if the app user owns a credential for this guild."""

    @abstractmethod
    def list_user_discord_guild_ids(self, user_id: int) -> List[str]:
        """Return distinct guild IDs connected by the given app user."""

    # ------------------------------------------------------------------
    # Telegram Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_telegram_credentials(self, phone_number: str) -> Optional[TelegramCredentialRow]:
        """Return the Telegram credentials for a phone number, or None."""

    @abstractmethod
    def save_telegram_credentials(
        self,
        user_id: int,
        phone_number: str,
        *,
        telegram_user_id: Optional[int] = None,
        session_string: str,
    ) -> TelegramCredentialRow:
        """Insert or update Telegram credentials for a phone number."""

    @abstractmethod
    def delete_telegram_credentials(self, phone_number: str) -> bool:
        """Delete Telegram credentials. Returns True if deleted."""

    # ------------------------------------------------------------------
    # Cleaned Messages
    # ------------------------------------------------------------------

    @abstractmethod
    def save_cleaned_messages(
        self,
        messages: List[Dict[str, Any]],
        provider: str = "slack",
    ) -> int:
        """Save a batch of cleaned messages. Returns count saved."""

    @abstractmethod
    def get_stored_messages(
        self,
        channel_id: Optional[str] = None,
        provider: str = "slack",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored cleaned messages."""

    # ------------------------------------------------------------------
    # Gmail auto-responder settings, watch/sync state, reply states
    # ------------------------------------------------------------------

    @abstractmethod
    def get_autoresponder_enabled(self, user_id: int) -> bool:
        """Return the user's auto-responder opt-in (False if the user is unknown)."""

    @abstractmethod
    def set_autoresponder_enabled(self, user_id: int, enabled: bool) -> None:
        """Persist the user's auto-responder opt-in."""

    @abstractmethod
    def find_gmail_sync_state(self, user_id: int) -> Optional[GmailSyncStateRow]:
        """Return the Gmail sync state for a user, or None."""

    @abstractmethod
    def find_gmail_sync_state_by_email(self, email_address: str) -> Optional[GmailSyncStateRow]:
        """Return the Gmail sync state for a mailbox address, or None."""

    @abstractmethod
    def list_gmail_sync_states(self) -> List[GmailSyncStateRow]:
        """Return all Gmail sync states."""

    @abstractmethod
    def save_gmail_watch(
        self,
        user_id: int,
        *,
        email_address: str,
        history_id: int,
        expiration: Optional[datetime],
        reset_history: bool = False,
    ) -> GmailSyncStateRow:
        """Record an active watch.

        `history_id` only seeds a missing sync pointer unless `reset_history`
        is set, in which case sync restarts from it.
        """

    @abstractmethod
    def update_gmail_sync_state(self, user_id: int, **fields: Any) -> None:
        """Update arbitrary columns of a user's Gmail sync state."""

    @abstractmethod
    def advance_gmail_history_id(self, user_id: int, history_id: int) -> bool:
        """Move the sync pointer forward only; returns True if it moved."""

    @abstractmethod
    def claim_gmail_message(
        self,
        user_id: int,
        gmail_message_id: str,
        *,
        lock_timeout: timedelta,
        max_attempts: int,
    ) -> Optional[GmailReplyStateRow]:
        """Atomically claim a Gmail message for processing.

        Returns the claimed row (with a fresh `lock_token`), or None when the
        message is finished, out of attempts, or owned by another worker.
        """

    @abstractmethod
    def update_gmail_reply_state(
        self,
        user_id: int,
        gmail_message_id: str,
        lock_token: str,
        *,
        release: bool = False,
        **fields: Any,
    ) -> bool:
        """Update a claimed reply state. Only the lock holder may write."""

    @abstractmethod
    def list_gmail_reply_work(
        self,
        user_id: int,
        *,
        lock_timeout: timedelta,
        max_attempts: int,
        limit: int = 20,
    ) -> List[str]:
        """Return Gmail message IDs with retryable unfinished work."""

    @abstractmethod
    def list_gmail_reply_states(self, user_id: int, limit: int = 10) -> List[GmailReplyStateRow]:
        """Return the user's most recently updated reply states."""
