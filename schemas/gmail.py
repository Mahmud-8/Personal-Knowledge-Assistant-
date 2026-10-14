"""
schemas/gmail.py
Request/response schemas for Gmail push notifications and the auto-responder.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AutoresponderSettingsRequest(BaseModel):
    """Turn automatic Gmail replies on or off."""
    enabled: bool = Field(..., description="Whether the AI may reply to incoming email.")


class GmailWatchStatus(BaseModel):
    """State of the Gmail push-notification watch and history sync."""
    status: str
    email_address: str
    expires_at: datetime | None = None
    last_notification_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_error: str | None = None


class GmailActivityItem(BaseModel):
    """Outcome of processing one incoming email (no internal IDs)."""
    sender: str | None = None
    subject: str | None = None
    status: str
    reply_sent: bool
    indexed: bool
    error: str | None = None
    updated_at: datetime | None = None


class GmailAutomationStatus(BaseModel):
    """Everything the UI shows about Gmail notifications and auto-replies."""
    google_connected: bool
    push_configured: bool
    autoresponder_enabled: bool
    watch: GmailWatchStatus | None = None
    recent_activity: list[GmailActivityItem] = Field(default_factory=list)
