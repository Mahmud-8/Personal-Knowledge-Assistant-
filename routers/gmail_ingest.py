"""
routes/ingest.py

Endpoint to trigger the RAG ingestion pipeline for a user's emails.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from loaders.gmail_loader import GmailLoader
from loaders.pdf_loader import PDFLoader
from initializer import db, embedder, store, dispatcher, chunker
from services.app_auth_service import get_current_user
from rag.pipeline import Pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])

# In-memory ingestion progress tracker keyed by user_id
_ingest_status: dict[int, dict] = {}


def _run_ingest(
    user_id: int,
    google_creds: dict,
    max_emails: int,
    label: str,
) -> None:
    """Background task that loads new emails and runs the pipeline."""
    _ingest_status[user_id] = {
        "state": "fetching",
        "fetched": 0,
        "new_emails": 0,
        "chunks_stored": 0,
    }

    try:
        loader = GmailLoader(
            access_token=google_creds["access_token"],
            refresh_token=google_creds["refresh_token"],
            max_emails=max_emails,
            label=label,
        )

        with loader:
            emails = loader.fetch_emails()

        _ingest_status[user_id]["fetched"] = len(emails)
        _ingest_status[user_id]["state"] = "processing"

        # Keep only emails that are not already present in Chroma for this
        # user; tagging the owner keeps one user's mail out of another's searches.
        new_emails = [
            email.model_copy(update={"owner_user_id": user_id})
            for email in emails
            if not store.has_email(email.id, owner_user_id=user_id)
        ]

        _ingest_status[user_id]["new_emails"] = len(new_emails)

        logger.info(
            "Detected %d new emails out of %d fetched.",
            len(new_emails),
            len(emails),
        )

        pipeline = Pipeline(
            store=store,
            embedder=embedder,
            dispatcher=dispatcher,
            chunker=chunker,
        )

        count = pipeline.run(new_emails)

        _ingest_status[user_id]["chunks_stored"] = count
        _ingest_status[user_id]["state"] = "done"

        logger.info(
            "Ingestion complete for user %d: %d new emails, %d chunks stored.",
            user_id,
            len(new_emails),
            count,
        )

    except Exception as exc:
        _ingest_status[user_id]["state"] = "error"
        _ingest_status[user_id]["error"] = str(exc)
        logger.exception(
            "Ingestion failed for user %d: %s",
            user_id,
            exc,
        )


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger email ingestion",
    description=(
        "Fetches emails for the given user and runs the RAG pipeline "
        "for new emails in the background."
    ),
)
def ingest(
    background_tasks: BackgroundTasks,
    max_emails: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Max emails to fetch",
    ),
    label: str = Query(
        default="INBOX",
        description="Gmail label to fetch from",
    ),
    user=Depends(get_current_user),
):
    """Fetch emails for a user and run the RAG pipeline in the background."""
    google_creds = db.find_google_credentials(user.id)

    if not google_creds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account not linked. Connect Google first.",
        )

    background_tasks.add_task(
        _run_ingest,
        user.id,
        {
            "access_token": google_creds.access_token,
            "refresh_token": google_creds.refresh_token,
        },
        max_emails,
        label,
    )

    return {
        "status": "Ingestion started",
        "max_emails": max_emails,
        "label": label,
    }


@router.get(
    "/ingest/status",
    summary="Check ingestion progress",
    description="Returns the current ingestion status for the logged-in user.",
)
def ingest_status(
    user=Depends(get_current_user),
):
    """Return the current ingestion progress for the logged-in user."""
    status_data = _ingest_status.get(user.id)

    if not status_data:
        return {"state": "idle"}

    return status_data


def _run_pdf_ingest(pdf_folder: str, cleanup: bool = True) -> None:
    """Background task that loads PDFs and runs the pipeline."""
    try:
        loader = PDFLoader(pdf_folder)
        documents = loader.load_pdfs()

        pipeline = Pipeline(
            store=store,
            embedder=embedder,
            dispatcher=dispatcher,
            chunker=chunker,
        )

        count = pipeline.run(documents)

        logger.info(
            "PDF ingestion complete: %d chunks stored from %s.",
            count,
            pdf_folder,
        )

    except Exception as exc:
        logger.exception(
            "PDF ingestion failed: %s",
            exc,
        )

    finally:
        if cleanup:
            shutil.rmtree(
                pdf_folder,
                ignore_errors=True,
            )


@router.post(
    "/ingest/pdf",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest PDF files",
    description="Uploads PDF files and runs the RAG pipeline in the background.",
)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(
        ...,
        description="PDF files to ingest",
    ),
    user=Depends(get_current_user),
):
    pdf_files = [
        f
        for f in files
        if f.filename and f.filename.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF files provided.",
        )

    tmp_dir = tempfile.mkdtemp(prefix="pdf_ingest_")

    for f in pdf_files:
        path = os.path.join(
            tmp_dir,
            f.filename,
        )

        with open(path, "wb") as out:
            content = await f.read()
            out.write(content)

    background_tasks.add_task(
        _run_pdf_ingest,
        tmp_dir,
        cleanup=True,
    )

    return {
        "status": "PDF ingestion started",
        "files": [f.filename for f in pdf_files],
    }