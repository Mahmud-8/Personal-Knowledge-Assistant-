"""
Personal Knowledge Assistant — a FastAPI application that ingests data from multiple platforms (Gmail, Slack, Discord, Telegram, PDFs), 
processes it through a RAG pipeline, 
and provides an agentic chat interface for semantic search over ingested content.
FastAPI application entry point. Configures the app, registers
routers, and runs database setup on startup via the lifespan handler.
"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from initializer import db

from routers import app_auth, base_route, chat, discord_auth, discord_ingest, gmail_auth, gmail_ingest, gmail_push, slack_auth, slack_ingest, telegram_auth, telegram_ingest
from services import gmail_push_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _gmail_maintenance_loop() -> None:
    """Renew Gmail watches before they expire and retry unfinished email work.

    New mail is delivered by Pub/Sub push; this loop does not poll for it.
    """
    while True:
        try:
            await asyncio.to_thread(gmail_push_service.run_maintenance)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Gmail maintenance failed.")
        await asyncio.sleep(settings.GMAIL_WATCH_RENEW_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """"STARTUP: runs once before the server accepts requests"""
    db.create_database()
    db.create_tables()
    maintenance = asyncio.create_task(_gmail_maintenance_loop())

    yield

    maintenance.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await maintenance

    # SHUTDOWN: runs once after the server stops accepting requests
    # (e.g., close connections, flush caches)

app = FastAPI(lifespan=lifespan)
app.include_router(base_route.base_router)
app.include_router(app_auth.router)
app.include_router(gmail_auth.router)
app.include_router(gmail_ingest.router)
app.include_router(gmail_push.router)
app.include_router(slack_auth.router)
app.include_router(slack_ingest.router)
app.include_router(discord_auth.router)
app.include_router(discord_ingest.router)
app.include_router(telegram_auth.router)
app.include_router(telegram_ingest.router)
app.include_router(chat.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
