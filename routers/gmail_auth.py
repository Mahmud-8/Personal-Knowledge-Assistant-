"""
routes/auth.py

HTTP layer for the Google OAuth flow. Routes only translate between HTTP and
AuthService — all business logic lives in services/auth_service.py.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from config import settings
from initializer import db
from services import gmail_push_service
from services.app_auth_service import get_current_user
from services.gmail_auth_service import (
    AuthError,
    AuthService,
    OAuthExchangeError,
    StateValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_service = AuthService()


@router.get(
    "/google/login",
    summary="Start Gmail sign-in",
    description="Redirects the browser to the Google OAuth consent screen. Requires JWT token.",
)
def google_login(
    user=Depends(get_current_user),
):
    """Return the Google OAuth authorization URL (frontend navigates to it)."""
    authorization_url, _ = _service.get_authorization_url(user_id=user.id)
    return {"authorization_url": authorization_url}


@router.get(
    "/google/callback",
    summary="Handle Google OAuth callback",
    description=(
        "Exchanges the authorization code for tokens, persists the user, and "
        "redirects to the homepage."
    ),
)
def google_callback(
    code: Optional[str] = Query(default=None, min_length=1),
    state: Optional[str] = Query(default=None, min_length=1),
    error: Optional[str] = Query(default=None),
):
    """Exchange authorization code for tokens and redirect to homepage."""
    if error:
        return RedirectResponse(
            url="/?google_error=" + urllib.parse.quote(f"Google OAuth error: {error}"),
            status_code=status.HTTP_302_FOUND,
        )

    if not code or not state:
        return RedirectResponse(
            url="/?google_error=" + urllib.parse.quote("Missing required query parameters."),
            status_code=status.HTTP_302_FOUND,
        )

    try:
        user = _service.authenticate(code, state)
    except (StateValidationError, OAuthExchangeError, AuthError) as exc:
        logger.exception("Google authentication failed.")
        return RedirectResponse(
            url="/?google_error=" + urllib.parse.quote(str(exc)),
            status_code=status.HTTP_302_FOUND,
        )

    # Start Gmail push notifications for the newly linked mailbox. Failure
    # is not fatal: the UI shows the watch status and can retry.
    if settings.gmail_push_configured:
        try:
            gmail_push_service.ensure_watch(user.id)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Starting Gmail watch after Google connect failed.")

    return RedirectResponse(url="/?google=connected", status_code=status.HTTP_302_FOUND)


@router.get(
    "/google/status",
    summary="Check if Google account is linked",
)
def google_status(
    user=Depends(get_current_user),
):
    """Check if the logged-in user has linked a Google account."""
    google_creds = db.find_google_credentials(user.id)
    return {"connected": google_creds is not None}
