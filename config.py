"""
Application settings loaded from environment variables / `.env`.
This module is configuration-only: no business logic, no HTTP, no DB.
"""

import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    First try to Load settings from OS environment variables and if failed
    then Loads settings from .env file and if failed 
    then loads them from the classdefault values.
    Loading from .env file is for development purposes only and should not be used in production.
    In production, all settings should be loaded from environment variables .
    Environment variables is set by the deployment environment (e.g. Docker, systemd, or your cloud provider's secret manager).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application OAuth
    SECRET_KEY: SecretStr
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Google OAuth 2.0
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: SecretStr
    # Must be EXACTLY one of the "Authorized redirect URIs" registered for the
    # OAuth client in Google Cloud Console (e.g. /api/v1/auth/google/callback).
    GOOGLE_REDIRECT_URI: str
    # Comma-separated list of OAuth scopes.
    GOOGLE_SCOPES: str = (
        "openid,email,profile,"
        "https://www.googleapis.com/auth/gmail.readonly,"
        "https://www.googleapis.com/auth/gmail.send"
    )
    # Gmail push notifications (Cloud Pub/Sub). See README "Gmail push
    # notifications" for the Google Cloud setup these values come from.
    # Full topic name: projects/<project-id>/topics/<topic>. Empty disables watch.
    GMAIL_PUBSUB_TOPIC: str = ""
    # OIDC push authentication: the audience configured on the push
    # subscription and the service account it signs tokens as.
    GMAIL_PUBSUB_AUDIENCE: str = ""
    GMAIL_PUBSUB_SERVICE_ACCOUNT: str = ""
    # Optional shared secret appended to the push endpoint as ?token=...
    GMAIL_PUBSUB_VERIFICATION_TOKEN: SecretStr = SecretStr("")
    # Optional full subscription name (projects/<id>/subscriptions/<name>);
    # when set, pushes from any other subscription are rejected.
    GMAIL_PUBSUB_SUBSCRIPTION: str = ""
    # Comma-separated Gmail label IDs that trigger notifications.
    GMAIL_WATCH_LABEL_IDS: str = "INBOX"
    # How often the renewal task runs, and how close to expiry a watch is renewed.
    GMAIL_WATCH_RENEW_INTERVAL_SECONDS: int = 6 * 60 * 60
    GMAIL_WATCH_RENEW_BEFORE_SECONDS: int = 24 * 60 * 60
    # A claimed email whose processing has not finished within this time
    # is considered abandoned and may be claimed again.
    GMAIL_REPLY_LOCK_TIMEOUT_SECONDS: int = 15 * 60
    GMAIL_REPLY_MAX_ATTEMPTS: int = 3

    # Groq / RAG
    GROQ_API_KEY: SecretStr
    GROQ_MODEL: str = "qwen/qwen3.6-27b"
    VECTOR_STORE_DIR: str

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_POOL_SIZE: int = 5

    # Discord OAuth 2.0
    DISCORD_CLIENT_ID: str = ""
    DISCORD_CLIENT_SECRET: SecretStr = SecretStr("")
    DISCORD_REDIRECT_URI: str = ""
    DISCORD_BOT_TOKEN: SecretStr = SecretStr("")
    DISCORD_BOT_SCOPES: str = "identify bot"
    DISCORD_BOT_PERMISSIONS: int = 8515702525261888

    # Slack OAuth 2.0
    SLACK_CLIENT_ID: str = ""
    SLACK_CLIENT_SECRET: SecretStr = SecretStr("")
    SLACK_REDIRECT_URI: str = ""
    SLACK_BOT_SCOPES: str = "channels:history,channels:read,groups:history,groups:read,im:history,im:read,mpim:history,mpim:read,users:read"
    SLACK_USER_SCOPES: str = ""
    SLACK_BOT_TOKEN: SecretStr = SecretStr("")

    # Telegram (Telethon MTProto)
    TELEGRAM_API_ID: int = 0
    TELEGRAM_API_HASH: SecretStr = SecretStr("")

    # HuggingFace
    HF_HUB_DISABLE_XET: str = "1"
    HF_HUB_OFFLINE: str = "1"

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection string for PostgreSQL."""
        from urllib.parse import quote_plus
        password = quote_plus(self.POSTGRES_PASSWORD.get_secret_value())
        user = quote_plus(self.POSTGRES_USER)
        return (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def google_scopes(self) -> list[str]:
        """Parse comma-separated scopes into a list."""
        return [scope.strip() for scope in self.GOOGLE_SCOPES.split(",") if scope.strip()]

    @property
    def gmail_watch_label_ids(self) -> list[str]:
        """Parse comma-separated Gmail watch label IDs into a list."""
        return [label.strip() for label in self.GMAIL_WATCH_LABEL_IDS.split(",") if label.strip()]

    @property
    def gmail_push_configured(self) -> bool:
        """True when push notifications can be both requested and verified."""
        has_auth = bool(
            (self.GMAIL_PUBSUB_AUDIENCE and self.GMAIL_PUBSUB_SERVICE_ACCOUNT)
            or self.GMAIL_PUBSUB_VERIFICATION_TOKEN.get_secret_value()
        )
        return bool(self.GMAIL_PUBSUB_TOPIC) and has_auth

    @property
    def google_client_config(self) -> dict:
        """Dict in the shape google_auth_oauthlib expects for a Web client."""
        return {
            "web": {
                "client_id": self.GOOGLE_CLIENT_ID,
                "client_secret": self.GOOGLE_CLIENT_SECRET.get_secret_value(),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [self.GOOGLE_REDIRECT_URI],
            }
        }


settings = Settings()
# Set environment variables for HuggingFace Hub based on settings
os.environ.setdefault("HF_HUB_DISABLE_XET", settings.HF_HUB_DISABLE_XET)
os.environ.setdefault("HF_HUB_OFFLINE", settings.HF_HUB_OFFLINE)
