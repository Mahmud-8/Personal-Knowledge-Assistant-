"""
SQLAlchemy ORM models for the application database.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, false, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # Opt-in: the AI only replies to incoming Gmail when this is True.
    autoresponder_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    google_credentials: Mapped[GoogleCredentialRow | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    slack_credentials: Mapped[list[SlackCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    discord_credentials: Mapped[list[DiscordCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    telegram_credentials: Mapped[list[TelegramCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class CredentialRow(Base):
    """Abstract base for OAuth credential tables — shared columns only."""

    __abstract__ = True

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class GoogleCredentialRow(CredentialRow):
    __tablename__ = "google_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    google_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    user: Mapped[User] = relationship(back_populates="google_credentials")


class SlackCredentialRow(CredentialRow):
    """Stores Slack OAuth tokens linked to a workspace (team_id)."""

    __tablename__ = "slack_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    team_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    bot_user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    user: Mapped[User] = relationship(back_populates="slack_credentials")


class DiscordCredentialRow(CredentialRow):
    """Stores Discord OAuth tokens per user+guild connection."""

    __tablename__ = "discord_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    discord_user_id: Mapped[str] = mapped_column(String(32), nullable=False)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False)
    token_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Bearer")
    scopes: Mapped[str] = mapped_column(Text, nullable=False, default="identify bot")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    granted_permissions: Mapped[str | None] = mapped_column(String(32), nullable=True)

    user: Mapped[User] = relationship(back_populates="discord_credentials")

    __table_args__ = (
        UniqueConstraint("discord_user_id", "guild_id", name="uq_discord_user_guild"),
    )


class TelegramCredentialRow(Base):
    """Stores Telegram session strings linked to a phone number."""

    __tablename__ = "telegram_credentials"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    phone_number: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    telegram_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    session_string: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    user: Mapped[User] = relationship(back_populates="telegram_credentials")


class CleanedMessageRow(Base):
    """Stores cleaned messages ready for RAG chunking and embedding."""

    __tablename__ = "cleaned_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False, default="slack")
    channel_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_boilerplate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_duplicate_of: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedup_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# BIGINT ids that still autoincrement on SQLite (used by the test suite).
_BigIntId = BigInteger().with_variant(Integer, "sqlite")


class GmailSyncStateRow(Base):
    """Per-user Gmail push-notification (users.watch) and history sync state."""

    __tablename__ = "gmail_sync_states"

    id: Mapped[int] = mapped_column(_BigIntId, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    # Mailbox address Gmail puts in every notification; maps it to the user.
    email_address: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Last Gmail historyId fully processed; history.list starts from here.
    history_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # "active", "stopped" or "error".
    watch_status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    watch_expiration: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_notification_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class GmailReplyStateRow(Base):
    """Processing state of one Gmail message for the auto-responder.

    One row per (user, Gmail message). `locked_at` is a processing lease:
    a worker owns the message while it is set and younger than the lock
    timeout, so retried or concurrent notifications cannot reply twice.
    """

    __tablename__ = "gmail_reply_states"

    id: Mapped[int] = mapped_column(_BigIntId, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    gmail_message_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    sender: Mapped[str | None] = mapped_column(Text, nullable=True)
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    # new | replied | failed | skipped_owner | skipped_automated |
    # skipped_disabled | skipped_manual_reply | skipped_resync | skipped_not_inbox | invalid
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="new")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lock_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    incoming_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reply_message_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Retrying a failed send is only safe when no email can have gone out.
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("user_id", "gmail_message_id", name="uq_gmail_reply_user_message"),
    )
