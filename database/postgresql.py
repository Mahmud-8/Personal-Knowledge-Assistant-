"""
PostgreSQL data-access layer using SQLAlchemy ORM.

Responsibilities:
  - Initial setup: create the application database and all ORM tables.
  - CRUD: insert, query, and update user rows (OAuth credentials, profile).

All methods use SQLAlchemy sessions managed via ``sessionmaker``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import sql
from sqlalchemy import Engine, and_, create_engine, delete, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from .base_database import BaseDatabase
from .models import (
    Base,
    CleanedMessageRow,
    DiscordCredentialRow,
    GmailReplyStateRow,
    GmailSyncStateRow,
    GoogleCredentialRow,
    SlackCredentialRow,
    TelegramCredentialRow,
    User,
)

logger = logging.getLogger(__name__)

# Idempotent schema changes for tables that already existed before a column
# was added; `create_all` only creates missing tables, never missing columns.
_MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
    "autoresponder_enabled BOOLEAN NOT NULL DEFAULT FALSE",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database(BaseDatabase):
    """PostgreSQL implementation of the database access layer."""

    def __init__(self, settings: Settings, engine: Optional[Engine] = None) -> None:
        self._settings = settings
        self._engine = engine or create_engine(
            settings.database_url,
            pool_size=settings.POSTGRES_POOL_SIZE,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    def _session(self) -> Session:
        return self._session_factory()

    def create_database(self) -> None:
        """Create the application database if it does not exist."""
        conn = psycopg2.connect(
            host=self._settings.POSTGRES_HOST,
            port=self._settings.POSTGRES_PORT,
            dbname="postgres",
            user=self._settings.POSTGRES_USER,
            password=self._settings.POSTGRES_PASSWORD.get_secret_value(),
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname=%s",
                    (self._settings.POSTGRES_DB,),
                )
                if not cur.fetchone():
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(self._settings.POSTGRES_DB)
                        )
                    )
                    logger.info("Database %s created", self._settings.POSTGRES_DB)
                else:
                    logger.info("Database %s already exists", self._settings.POSTGRES_DB)
        finally:
            conn.close()

    def create_tables(self) -> None:
        """Create all ORM-defined tables if they do not already exist."""
        Base.metadata.create_all(self._engine)
        with self._engine.begin() as conn:
            for statement in _MIGRATIONS:
                conn.execute(text(statement))

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def find_user_by_id(self, user_id: int) -> Optional[User]:
        with self._session() as session:
            user = session.get(User, user_id)
            if user:
                session.expunge(user)
            return user

    def user_exists(self, email: str) -> bool:
        with self._session() as session:
            stmt = select(User.id).where(User.email == email)
            return session.execute(stmt).first() is not None

    def find_user_by_email(self, email: str) -> Optional[User]:
        with self._session() as session:
            stmt = select(User).where(User.email == email)
            user = session.scalars(stmt).first()
            if user:
                session.expunge(user)
            return user

    def create_user(self, *, name: str, email: str, password_hash: str) -> User:
        with self._session() as session:
            user = User(name=name, email=email, password_hash=password_hash)
            session.add(user)
            session.commit()
            session.refresh(user)
            session.expunge(user)
            return user

    def update_user_profile(self, user_id: int, *, name: str) -> User:
        with self._session() as session:
            user = session.get(User, user_id)
            user.name = name
            session.commit()
            session.refresh(user)
            session.expunge(user)
            return user

    # ------------------------------------------------------------------
    # Google Credentials
    # ------------------------------------------------------------------

    def find_google_credentials(self, user_id: int) -> Optional[GoogleCredentialRow]:
        with self._session() as session:
            stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

    def find_user_by_google_id(self, google_id: str) -> Optional[GoogleCredentialRow]:
        with self._session() as session:
            stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.google_id == google_id)
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

    def save_google_credentials(
        self,
        user_id: int,
        *,
        google_id: str,
        access_token: str,
        refresh_token: Optional[str],
        token_expiry: Optional[datetime],
    ) -> GoogleCredentialRow:
        with self._session() as session:
            # Check if this Google account is already linked to a different user
            stmt = select(GoogleCredentialRow).where(
                GoogleCredentialRow.google_id == google_id,
                GoogleCredentialRow.user_id != user_id,
            )
            if session.scalars(stmt).first():
                raise ValueError("This Google account is already linked to another user.")

            # Upsert: find existing row for user or create new
            stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)
            cred = session.scalars(stmt).first()

            if cred:
                cred.google_id = google_id
                cred.access_token = access_token
                if refresh_token is not None:
                    cred.refresh_token = refresh_token
                cred.token_expiry = token_expiry
            else:
                cred = GoogleCredentialRow(
                    user_id=user_id,
                    google_id=google_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_expiry=token_expiry,
                )
                session.add(cred)

            session.commit()
            session.refresh(cred)
            session.expunge(cred)
            return cred

    # ------------------------------------------------------------------
    # Slack Credentials
    # ------------------------------------------------------------------

    def find_slack_credentials(self, team_id: str) -> Optional[SlackCredentialRow]:
        with self._session() as session:
            stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

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
        with self._session() as session:
            stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
            cred = session.scalars(stmt).first()

            meta_str = json.dumps(metadata) if metadata else None

            if cred:
                cred.user_id = user_id
                cred.access_token = access_token
                if refresh_token is not None:
                    cred.refresh_token = refresh_token
                if team_name is not None:
                    cred.team_name = team_name
                if bot_user_id is not None:
                    cred.bot_user_id = bot_user_id
                if meta_str is not None:
                    cred.metadata_json = meta_str
            else:
                cred = SlackCredentialRow(
                    user_id=user_id,
                    team_id=team_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    team_name=team_name,
                    bot_user_id=bot_user_id,
                    metadata_json=meta_str,
                )
                session.add(cred)

            session.commit()
            session.refresh(cred)
            session.expunge(cred)
            return cred

    def delete_slack_credentials(self, team_id: str) -> bool:
        with self._session() as session:
            stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
            cred = session.scalars(stmt).first()
            if cred:
                session.delete(cred)
                session.commit()
                return True
            return False

    # ------------------------------------------------------------------
    # Discord Credentials
    # ------------------------------------------------------------------

    def find_discord_credentials(
        self, discord_user_id: str, guild_id: str
    ) -> Optional[DiscordCredentialRow]:
        with self._session() as session:
            stmt = select(DiscordCredentialRow).where(
                DiscordCredentialRow.discord_user_id == discord_user_id,
                DiscordCredentialRow.guild_id == guild_id,
            )
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

    def find_discord_credentials_by_guild(self, guild_id: str) -> Optional[DiscordCredentialRow]:
        with self._session() as session:
            stmt = select(DiscordCredentialRow).where(
                DiscordCredentialRow.guild_id == guild_id
            )
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

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
        with self._session() as session:
            stmt = select(DiscordCredentialRow).where(
                DiscordCredentialRow.discord_user_id == discord_user_id,
                DiscordCredentialRow.guild_id == guild_id,
            )
            cred = session.scalars(stmt).first()

            if cred:
                cred.user_id = user_id
                cred.access_token = access_token
                if refresh_token is not None:
                    cred.refresh_token = refresh_token
                cred.token_type = token_type
                cred.scopes = scopes
                if expires_at is not None:
                    cred.expires_at = expires_at
                cred.granted_permissions = granted_permissions
            else:
                cred = DiscordCredentialRow(
                    user_id=user_id,
                    discord_user_id=discord_user_id,
                    guild_id=guild_id,
                    access_token=access_token,
                    refresh_token=refresh_token,
                    token_type=token_type,
                    scopes=scopes,
                    expires_at=expires_at,
                    granted_permissions=granted_permissions,
                )
                session.add(cred)

            session.commit()
            session.refresh(cred)
            session.expunge(cred)
            return cred

    def delete_discord_credentials(self, discord_user_id: str, guild_id: str) -> bool:
        with self._session() as session:
            stmt = select(DiscordCredentialRow).where(
                DiscordCredentialRow.discord_user_id == discord_user_id,
                DiscordCredentialRow.guild_id == guild_id,
            )
            cred = session.scalars(stmt).first()
            if cred:
                session.delete(cred)
                session.commit()
                return True
            return False

    def list_discord_guild_ids(self) -> List[str]:
        with self._session() as session:
            return list(session.scalars(select(DiscordCredentialRow.guild_id).distinct()))

    def discord_guild_connection_exists(self, guild_id: str) -> bool:
        with self._session() as session:
            stmt = select(DiscordCredentialRow.id).where(
                DiscordCredentialRow.guild_id == guild_id
            ).limit(1)
            return session.execute(stmt).first() is not None

    def discord_guild_owned_by_user(self, user_id: int, guild_id: str) -> bool:
        with self._session() as session:
            stmt = select(DiscordCredentialRow.id).where(
                DiscordCredentialRow.user_id == user_id,
                DiscordCredentialRow.guild_id == guild_id,
            ).limit(1)
            return session.execute(stmt).first() is not None

    def list_user_discord_guild_ids(self, user_id: int) -> List[str]:
        with self._session() as session:
            stmt = select(DiscordCredentialRow.guild_id).where(
                DiscordCredentialRow.user_id == user_id
            ).distinct()
            return list(session.scalars(stmt))

    # ------------------------------------------------------------------
    # Telegram Credentials
    # ------------------------------------------------------------------

    def find_telegram_credentials(self, phone_number: str) -> Optional[TelegramCredentialRow]:
        with self._session() as session:
            stmt = select(TelegramCredentialRow).where(
                TelegramCredentialRow.phone_number == phone_number
            )
            cred = session.scalars(stmt).first()
            if cred:
                session.expunge(cred)
            return cred

    def save_telegram_credentials(
        self,
        user_id: int,
        phone_number: str,
        *,
        telegram_user_id: Optional[int] = None,
        session_string: str,
    ) -> TelegramCredentialRow:
        with self._session() as session:
            stmt = select(TelegramCredentialRow).where(
                TelegramCredentialRow.phone_number == phone_number
            )
            cred = session.scalars(stmt).first()

            if cred:
                cred.user_id = user_id
                cred.session_string = session_string
                if telegram_user_id is not None:
                    cred.telegram_user_id = telegram_user_id
            else:
                cred = TelegramCredentialRow(
                    user_id=user_id,
                    phone_number=phone_number,
                    telegram_user_id=telegram_user_id,
                    session_string=session_string,
                )
                session.add(cred)

            session.commit()
            session.refresh(cred)
            session.expunge(cred)
            return cred

    def delete_telegram_credentials(self, phone_number: str) -> bool:
        with self._session() as session:
            stmt = select(TelegramCredentialRow).where(
                TelegramCredentialRow.phone_number == phone_number
            )
            cred = session.scalars(stmt).first()
            if cred:
                session.delete(cred)
                session.commit()
                return True
            return False

    # ------------------------------------------------------------------
    # Cleaned Messages
    # ------------------------------------------------------------------

    def save_cleaned_messages(
        self,
        messages: List[Dict[str, Any]],
        provider: str = "slack",
    ) -> int:
        saved_count = 0
        with self._session() as session:
            for msg in messages:
                meta = msg.get("metadata") or {}
                record = CleanedMessageRow(
                    provider=provider,
                    channel_id=msg.get("channel_id") or meta.get("channel"),
                    message_id=msg.get("message_id") or meta.get("ts"),
                    user_id=msg.get("user_id") or meta.get("user"),
                    raw_text=msg.get("raw_text", ""),
                    cleaned_text=msg.get("cleaned_text", ""),
                    is_boilerplate=msg.get("is_boilerplate", False),
                    is_duplicate_of=msg.get("is_duplicate_of"),
                    dedup_hash=msg.get("dedup_hash"),
                    metadata_json=json.dumps(meta) if meta else None,
                )
                session.add(record)
                saved_count += 1
            session.commit()
        return saved_count

    def get_stored_messages(
        self,
        channel_id: Optional[str] = None,
        provider: str = "slack",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._session() as session:
            stmt = select(CleanedMessageRow).where(CleanedMessageRow.provider == provider)
            if channel_id:
                stmt = stmt.where(CleanedMessageRow.channel_id == channel_id)
            stmt = stmt.order_by(CleanedMessageRow.id.desc()).limit(limit)
            records = session.scalars(stmt).all()
            results = []
            for r in records:
                meta = {}
                if r.metadata_json:
                    try:
                        meta = json.loads(r.metadata_json)
                    except Exception:
                        pass
                results.append({
                    "id": r.id,
                    "provider": r.provider,
                    "channel_id": r.channel_id,
                    "message_id": r.message_id,
                    "user_id": r.user_id,
                    "raw_text": r.raw_text,
                    "cleaned_text": r.cleaned_text,
                    "is_boilerplate": r.is_boilerplate,
                    "is_duplicate_of": r.is_duplicate_of,
                    "dedup_hash": r.dedup_hash,
                    "metadata": meta,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                })
            return results

    # ------------------------------------------------------------------
    # Gmail auto-responder settings
    # ------------------------------------------------------------------

    def get_autoresponder_enabled(self, user_id: int) -> bool:
        with self._session() as session:
            stmt = select(User.autoresponder_enabled).where(User.id == user_id)
            return bool(session.scalars(stmt).first())

    def set_autoresponder_enabled(self, user_id: int, enabled: bool) -> None:
        with self._session() as session:
            session.execute(
                update(User).where(User.id == user_id).values(autoresponder_enabled=enabled)
            )
            session.commit()

    # ------------------------------------------------------------------
    # Gmail watch / history sync state
    # ------------------------------------------------------------------

    def find_gmail_sync_state(self, user_id: int) -> Optional[GmailSyncStateRow]:
        with self._session() as session:
            stmt = select(GmailSyncStateRow).where(GmailSyncStateRow.user_id == user_id)
            row = session.scalars(stmt).first()
            if row:
                session.expunge(row)
            return row

    def find_gmail_sync_state_by_email(self, email_address: str) -> Optional[GmailSyncStateRow]:
        with self._session() as session:
            stmt = select(GmailSyncStateRow).where(
                GmailSyncStateRow.email_address == email_address.strip().lower()
            )
            row = session.scalars(stmt).first()
            if row:
                session.expunge(row)
            return row

    def list_gmail_sync_states(self) -> List[GmailSyncStateRow]:
        with self._session() as session:
            rows = list(session.scalars(select(GmailSyncStateRow)))
            for row in rows:
                session.expunge(row)
            return rows

    def save_gmail_watch(
        self,
        user_id: int,
        *,
        email_address: str,
        history_id: int,
        expiration: Optional[datetime],
        reset_history: bool = False,
    ) -> GmailSyncStateRow:
        email_address = email_address.strip().lower()
        with self._session() as session:
            # A mailbox belongs to one user; drop a stale mapping left by a
            # previous link of the same Google account.
            session.execute(
                delete(GmailSyncStateRow).where(
                    GmailSyncStateRow.email_address == email_address,
                    GmailSyncStateRow.user_id != user_id,
                )
            )
            stmt = select(GmailSyncStateRow).where(GmailSyncStateRow.user_id == user_id)
            row = session.scalars(stmt).first()
            if row is None:
                row = GmailSyncStateRow(user_id=user_id, email_address=email_address)
                session.add(row)
            elif row.email_address != email_address:
                # A different mailbox was connected: its history starts fresh.
                row.email_address = email_address
                row.history_id = None

            # Renewing keeps the existing pointer so no message is skipped.
            if row.history_id is None or reset_history:
                row.history_id = history_id
            row.watch_status = "active"
            row.watch_expiration = expiration
            row.last_error = None

            session.commit()
            session.refresh(row)
            session.expunge(row)
            return row

    def update_gmail_sync_state(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        with self._session() as session:
            session.execute(
                update(GmailSyncStateRow)
                .where(GmailSyncStateRow.user_id == user_id)
                .values(**fields)
            )
            session.commit()

    def advance_gmail_history_id(self, user_id: int, history_id: int) -> bool:
        with self._session() as session:
            result = session.execute(
                update(GmailSyncStateRow)
                .where(
                    GmailSyncStateRow.user_id == user_id,
                    or_(
                        GmailSyncStateRow.history_id.is_(None),
                        GmailSyncStateRow.history_id < history_id,
                    ),
                )
                .values(history_id=history_id)
            )
            session.commit()
            return result.rowcount == 1

    # ------------------------------------------------------------------
    # Gmail reply states (processing lease + outcome per message)
    # ------------------------------------------------------------------

    @staticmethod
    def _gmail_reply_needs_work():
        """Rows whose reply decision or indexing is still unfinished."""
        row = GmailReplyStateRow
        return or_(
            and_(row.status.in_(("new", "failed")), row.retryable.is_(True)),
            row.incoming_indexed.is_(False),
            and_(row.reply_message_id.is_not(None), row.reply_indexed.is_(False)),
        )

    def claim_gmail_message(
        self,
        user_id: int,
        gmail_message_id: str,
        *,
        lock_timeout: timedelta,
        max_attempts: int,
    ) -> Optional[GmailReplyStateRow]:
        now = _utcnow()
        token = uuid.uuid4().hex
        row = GmailReplyStateRow

        with self._session() as session:
            try:
                # First sight of this message: the unique constraint makes
                # the insert itself the claim.
                session.add(GmailReplyStateRow(
                    user_id=user_id,
                    gmail_message_id=gmail_message_id,
                    status="new",
                    attempts=1,
                    locked_at=now,
                    lock_token=token,
                ))
                session.commit()
            except IntegrityError:
                session.rollback()
                # Existing row: a single conditional UPDATE is atomic, so at
                # most one worker can take the lease.
                result = session.execute(
                    update(row)
                    .where(
                        row.user_id == user_id,
                        row.gmail_message_id == gmail_message_id,
                        or_(row.locked_at.is_(None), row.locked_at < now - lock_timeout),
                        row.attempts < max_attempts,
                        self._gmail_reply_needs_work(),
                    )
                    .values(locked_at=now, lock_token=token, attempts=row.attempts + 1)
                    .execution_options(synchronize_session=False)
                )
                session.commit()
                if result.rowcount != 1:
                    return None

            claimed = session.scalars(
                select(row).where(
                    row.user_id == user_id,
                    row.gmail_message_id == gmail_message_id,
                    row.lock_token == token,
                )
            ).first()
            if claimed:
                session.expunge(claimed)
            return claimed

    def update_gmail_reply_state(
        self,
        user_id: int,
        gmail_message_id: str,
        lock_token: str,
        *,
        release: bool = False,
        **fields: Any,
    ) -> bool:
        if release:
            fields.update(locked_at=None, lock_token=None)
        if not fields:
            return True
        row = GmailReplyStateRow
        with self._session() as session:
            result = session.execute(
                update(row)
                .where(
                    row.user_id == user_id,
                    row.gmail_message_id == gmail_message_id,
                    row.lock_token == lock_token,
                )
                .values(**fields)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return result.rowcount == 1

    def list_gmail_reply_work(
        self,
        user_id: int,
        *,
        lock_timeout: timedelta,
        max_attempts: int,
        limit: int = 20,
    ) -> List[str]:
        row = GmailReplyStateRow
        cutoff = _utcnow() - lock_timeout
        with self._session() as session:
            stmt = (
                select(row.gmail_message_id)
                .where(
                    row.user_id == user_id,
                    or_(row.locked_at.is_(None), row.locked_at < cutoff),
                    row.attempts < max_attempts,
                    self._gmail_reply_needs_work(),
                )
                .order_by(row.id)
                .limit(limit)
            )
            return list(session.scalars(stmt))

    def list_gmail_reply_states(self, user_id: int, limit: int = 10) -> List[GmailReplyStateRow]:
        with self._session() as session:
            stmt = (
                select(GmailReplyStateRow)
                .where(GmailReplyStateRow.user_id == user_id)
                .order_by(GmailReplyStateRow.updated_at.desc(), GmailReplyStateRow.id.desc())
                .limit(limit)
            )
            rows = list(session.scalars(stmt))
            for r in rows:
                session.expunge(r)
            return rows
