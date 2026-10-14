# Personal Knowledge Assistant

A FastAPI application that ingests data from multiple platforms (Gmail, Slack, Discord, Telegram, PDFs), processes it through a RAG pipeline, and provides an agentic chat interface for semantic search over your personal knowledge base.

## Features

- **Multi-platform ingestion** — Gmail, Slack, Discord, Telegram, and PDF uploads
- **RAG pipeline** — cleaning, chunking, embedding, and vector storage
- **Vector-store search** — the agent's `search` tool queries ChromaDB, optionally filtered by source (`email`, `slack`, `discord`, `telegram`, `pdf`); email results are only returned to the user who owns them
- **Agentic chat** — conversational interface powered by LLM with tool-use capabilities
- **OAuth integration** — secure authentication flows for each platform
- **Gmail auto-responder** — new emails detected via Gmail push notifications (Pub/Sub), indexed, and optionally answered from your email history
- **Web UI** — single-page frontend served at `/` (register/login, connect accounts, ingest, chat, Gmail automation)

## Architecture

```
Loader → CleanerDispatcher → LangChainChunker → SentenceTransformerEmbedder → ChromaVectorStore
```

| Component | Description |
|-----------|-------------|
| **Loaders** | Fetch raw data from Gmail, Slack, Discord, Telegram, and PDFs |
| **Cleaners** | Normalize and sanitize content (HTML stripping, whitespace, emoji handling) |
| **Chunker** | Splits text into overlapping chunks (1000 chars, 200 overlap) |
| **Embedder** | Generates 768-dim vectors using `all-mpnet-base-v2` |
| **Vector Store** | Persists embeddings in ChromaDB with `source_type` filtering |
| **Agent** | Multi-turn tool-use loop against LLM for conversational search |
| **Database** | PostgreSQL (SQLAlchemy) for users, platform credentials, cleaned messages, and Gmail sync/reply state |

## Prerequisites

**Required**

- Python 3.12 (developed and tested with 3.12.2)
- PostgreSQL, with a user that can create databases (or an existing database — see [Database setup](#database-setup))
- A [Groq](https://console.groq.com/) API key
- A Google Cloud project with an OAuth client (Gmail is the core integration; `GOOGLE_*` settings are required for the app to start)
- Internet access on first start, to download the `all-mpnet-base-v2` embedding model

**Optional**

- Slack, Discord, and Telegram app credentials — only for those integrations
- Google Cloud Pub/Sub, the `gcloud` CLI, and a public HTTPS URL — only for Gmail push notifications and automatic replies

## Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   `psycopg2` is built from source on some platforms. If installation fails,
   install the PostgreSQL client development headers (e.g. `libpq-dev`), or
   replace it with `psycopg2-binary` for local development.

4. **Configure environment variables**

   Copy the template and fill in your own values:

   ```bash
   cp .env.example .env
   ```

   Settings are read from OS environment variables first, then from `.env`
   (see `config.py`). Never commit `.env`.

   | Variable | Required | Notes |
   |----------|----------|-------|
   | `SECRET_KEY` | Yes | Long random string used to sign JWTs |
   | `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` | No | Defaults `HS256`, `30` |
   | `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Yes | `POSTGRES_HOST`/`POSTGRES_PORT` default to `localhost`/`5432` |
   | `GROQ_API_KEY` | Yes | `GROQ_MODEL` defaults to `qwen/qwen3.6-27b` |
   | `VECTOR_STORE_DIR` | Yes | ChromaDB directory, e.g. `data/chroma` (git-ignored) |
   | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` | Yes | See [Google OAuth and Gmail API](#google-oauth-and-gmail-api) |
   | `HF_HUB_OFFLINE` | No | Defaults to `1` (offline). Set `0` until the embedding model has been downloaded once |
   | `GMAIL_PUBSUB_*`, `GMAIL_WATCH_*`, `GMAIL_REPLY_*` | No | Gmail push / auto-responder only |
   | `SLACK_*`, `DISCORD_*`, `TELEGRAM_*` | No | Only for those integrations |

5. **Run the server**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 5000
   ```

   - Web UI: `http://localhost:5000`
   - Interactive API docs (Swagger): `http://localhost:5000/docs`
   - Health check: `GET http://localhost:5000/api/v1/`

6. **Run the tests**

   ```bash
   pip install -r requirements-dev.txt
   python -m pytest tests
   ```

   Tests use SQLite, a temporary ChromaDB, and mocked Gmail/Pub/Sub/LLM. No real email is sent, and no PostgreSQL server, `.env`, or model download is needed.

## Database setup

There is no separate migration tool. On startup (`main.py` lifespan):

1. `create_database()` connects to the server's `postgres` database and creates `POSTGRES_DB` if it does not exist. The configured user needs the `CREATEDB` privilege, or you can create the database yourself first:

   ```bash
   createdb knowledge_assistant
   ```

2. `create_tables()` creates all ORM tables that do not exist yet (`users`, platform credential tables, `cleaned_messages`, `gmail_sync_states`, `gmail_reply_states`) and applies idempotent schema updates (e.g. adding `users.autoresponder_enabled`).

## Google OAuth and Gmail API

1. In [Google Cloud Console](https://console.cloud.google.com/), create or select a project and enable the **Gmail API**.
2. Configure the **OAuth consent screen**. While the app is in *Testing* status, add every Gmail account you will connect as a **test user**.
3. Create an **OAuth client ID** of type **Web application** and add this **Authorized redirect URI** (it must match `GOOGLE_REDIRECT_URI` exactly):

   ```
   http://localhost:5000/auth/google/callback
   ```

4. Put the client ID, client secret and redirect URI in `.env`.

The app requests these scopes (`GOOGLE_SCOPES`): `openid`, `email`, `profile`, `gmail.readonly`, and `gmail.send`. `gmail.send` is only used by the auto-responder. The OAuth flow requests offline access, so a refresh token is stored for background work.

## Using the app

1. Open `http://localhost:5000`, **Register**, then **Login**.
2. Click **Connect Google Account** and complete the Google consent screen. You are redirected back to the app.
3. Click **Start Ingestion** in the Gmail card to fetch and index emails (`POST /api/v1/ingest`, default 50 messages from `INBOX`). Use **Upload & Ingest** for PDFs.
4. Ask questions in **Chat with Agent**, e.g. "What did Alice say about the invoice?". The agent searches the vector store and answers from the results.
5. Optional: if Pub/Sub is configured, use **Enable Notifications** and **Reply automatically to new emails** in the Gmail card (see [below](#gmail-push-notifications--auto-responder)).

## API Endpoints

All endpoints except register/login, OAuth callbacks, health checks, and the Pub/Sub push endpoint require a JWT: `Authorization: Bearer <access_token>` from `/api/v1/auth/login`.

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register a new user |
| POST | `/api/v1/auth/login` | Login and receive JWT token |
| POST | `/api/v1/auth/logout` | Logout and revoke token |

### Platform Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/google/login` | Start Google OAuth flow |
| GET | `/auth/google/callback` | Google OAuth callback |
| GET | `/auth/google/status` | Check if a Google account is linked |
| GET | `/slack/install` | Start Slack OAuth flow (callback: `/slack/oauth/callback`) |
| GET | `/discord/authorize` | Start Discord OAuth flow (callback: `/discord/callback`) |
| POST | `/telegram/start` | Send Telegram login code |
| POST | `/telegram/verify` | Verify Telegram login code |

### Ingestion

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ingest` | Ingest emails from Gmail (`max_emails`, `label` query params) |
| GET | `/api/v1/ingest/status` | Gmail ingestion progress |
| POST | `/api/v1/ingest/slack` | Ingest messages from Slack |
| POST | `/api/v1/ingest/discord` | Ingest messages from Discord |
| POST | `/api/v1/ingest/telegram` | Ingest messages from Telegram |
| POST | `/api/v1/ingest/pdf` | Upload and ingest PDF files |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Send a message to the assistant |
| POST | `/api/v1/chat/reset` | Reset a chat session |

### Gmail Notifications & Auto-Responder

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/gmail/push` | Pub/Sub push endpoint (authenticated by Pub/Sub, not a user JWT) |
| GET | `/api/v1/gmail/automation` | Watch, sync, auto-responder setting and recent activity |
| PUT | `/api/v1/gmail/autoresponder` | Enable/disable automatic replies (`{"enabled": true}`) |
| POST | `/api/v1/gmail/watch` | Start or renew Gmail push notifications |
| DELETE | `/api/v1/gmail/watch` | Stop Gmail push notifications |

The full list, including status and listing endpoints, is at `/docs`.

## Gmail Push Notifications & Auto-Responder

*Optional.* Everything else works without it. It is required only for
detecting new mail automatically and for automatic replies.

New Gmail messages are detected through **Gmail push notifications via Google
Cloud Pub/Sub**. Nobody needs to enter a message ID, and the server does not
poll Gmail for new mail.

```
Gmail ──users.watch──▶ Pub/Sub topic ──push──▶ POST /api/v1/gmail/push
  → verify Pub/Sub token → map emailAddress to user → History API from stored historyId
  → for each new message: claim (gmail_reply_states) → fetch → index in ChromaDB
  → reply conditions → agent searches email history (source="email") → send via Gmail API
  → index the sent reply → advance historyId
```

**Reply conditions**, checked in this order. The incoming email is indexed
before any of them, so its history is kept whatever the outcome:

1. Messages from the mailbox owner (the `SENT` label, or the owner's address) are never answered.
2. Automated mail (`Auto-Submitted`, `Precedence: bulk/list`, `List-Id`/`List-Unsubscribe`, no-reply senders) is not answered. This prevents reply loops.
3. Replies are only sent when the user has turned on `autoresponder_enabled` (off by default). Otherwise the status is `skipped_disabled`.
4. If the owner has already replied in the thread after the incoming message, the AI does not reply, and the owner's reply is indexed.
5. Each message is claimed atomically in `gmail_reply_states` with a 15-minute processing lease. Duplicate or concurrent notifications therefore never send a second reply.

**Failures**: if the agent fails or Gmail rejects a send (API or OAuth error),
the email stays indexed and the message is retried, up to
`GMAIL_REPLY_MAX_ATTEMPTS` times. A transport error *during* a send might
have delivered the email, so it is never retried. Indexing failures are
retried without re-sending. If Gmail's history has expired (it keeps roughly
one week), messages received since the last good sync are indexed but **not**
auto-answered, because they may be stale.

**Watch renewal**: Gmail watches expire after 7 days. A background task
(every `GMAIL_WATCH_RENEW_INTERVAL_SECONDS`, default 6 h) renews watches that
expire within `GMAIL_WATCH_RENEW_BEFORE_SECONDS` (default 24 h). Each run also
catches up through the History API and retries unfinished work. This task is
not the mechanism for detecting new mail.

**Isolation**: email chunks in ChromaDB carry `owner_user_id`, and email
results are only returned to their owner. Chunks indexed before this change
have no owner tag and are no longer returned as email history. Re-run Gmail
ingestion for each user to re-index them under that user.

### Google Cloud setup (manual, one-time)

Push notifications only work after the following one-time setup. Use the
same Google Cloud project as the OAuth client (`GOOGLE_CLIENT_ID`). The
existing OAuth scopes (`gmail.readonly`, `gmail.send`) are enough, so
users do not need to reconnect.

1. Enable the **Gmail API** and **Cloud Pub/Sub API** in the project.
2. Create the topic, and allow Gmail to publish to it:

   ```bash
   gcloud pubsub topics create gmail-notifications
   gcloud pubsub topics add-iam-policy-binding gmail-notifications \
     --member=serviceAccount:gmail-api-push@system.gserviceaccount.com \
     --role=roles/pubsub.publisher
   ```

3. Create a service account that Pub/Sub uses to sign push requests (OIDC):

   ```bash
   gcloud iam service-accounts create gmail-push-invoker
   # Only needed for projects created before April 2021:
   gcloud iam service-accounts add-iam-policy-binding \
     gmail-push-invoker@PROJECT_ID.iam.gserviceaccount.com \
     --member=serviceAccount:service-PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com \
     --role=roles/iam.serviceAccountTokenCreator
   ```

4. Create the push subscription. The endpoint must be **public HTTPS**. For
   local development, use a tunnel such as `ngrok http 5000` or
   `cloudflared tunnel`. Temporary tunnel URLs change when the tunnel
   restarts; when that happens, update the subscription's push endpoint and
   audience and `GMAIL_PUBSUB_AUDIENCE` to the new URL.

   ```bash
   gcloud pubsub subscriptions create gmail-push \
     --topic=gmail-notifications \
     --push-endpoint=https://YOUR_HOST/api/v1/gmail/push \
     --push-auth-service-account=gmail-push-invoker@PROJECT_ID.iam.gserviceaccount.com \
     --push-auth-token-audience=https://YOUR_HOST/api/v1/gmail/push \
     --ack-deadline=60
   ```

5. Add to `.env` (see `.env.example`):

   ```env
   GMAIL_PUBSUB_TOPIC=projects/PROJECT_ID/topics/gmail-notifications
   GMAIL_PUBSUB_AUDIENCE=https://YOUR_HOST/api/v1/gmail/push
   GMAIL_PUBSUB_SERVICE_ACCOUNT=gmail-push-invoker@PROJECT_ID.iam.gserviceaccount.com
   GMAIL_PUBSUB_SUBSCRIPTION=projects/PROJECT_ID/subscriptions/gmail-push
   # Optional alternative/addition to OIDC: a shared secret. If you use it,
   # append ?token=<value> to the push endpoint URL in step 4.
   # GMAIL_PUBSUB_VERIFICATION_TOKEN=long-random-string
   ```

   If no push authentication is configured, the push endpoint rejects every
   request (HTTP 503).

6. Restart the server. On startup it creates the new tables
   (`gmail_sync_states`, `gmail_reply_states`) and adds
   `users.autoresponder_enabled`. Then, for each user, click **Enable
   Notifications** in the Gmail card (connecting Google also starts the
   watch automatically), and turn on **Reply automatically to new emails**
   if wanted.
7. Verify end to end: send an email to the connected mailbox from another
   account. The Gmail card's activity list should show it within seconds.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Startup fails with a pydantic `ValidationError` ("Field required") | A required variable is missing from `.env`. Compare with `.env.example`. |
| Startup fails loading `all-mpnet-base-v2` | The model is not cached and `HF_HUB_OFFLINE` is `1` (the default). Set `HF_HUB_OFFLINE=0` for the first start. |
| `psycopg2.OperationalError` on startup | PostgreSQL is not running or `POSTGRES_*` values are wrong. The app also connects to the `postgres` database to create `POSTGRES_DB`. |
| `permission denied to create database` | Grant `CREATEDB` to the user, or create `POSTGRES_DB` manually (see [Database setup](#database-setup)). |
| Google `Error 400: redirect_uri_mismatch` | `GOOGLE_REDIRECT_URI` must exactly match an Authorized redirect URI on the OAuth client (scheme, host, port, and path `/auth/google/callback`). |
| Google "Access blocked" / app not verified | Add the Gmail account as a test user on the OAuth consent screen. |
| Gmail calls fail after working earlier (token expired or revoked) | Reconnect with **Connect Google Account**. The flow forces consent, so a new refresh token is issued. Google expires refresh tokens after 7 days for apps in *Testing* status. |
| `401` from API endpoints | Missing or expired JWT. Log in again (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 30). |
| `POST /api/v1/gmail/watch` returns `400` | Google account is not linked. Connect Google first. |
| `POST /api/v1/gmail/watch` returns `503`, or push returns `503` | Pub/Sub is not configured: set `GMAIL_PUBSUB_TOPIC` plus `GMAIL_PUBSUB_AUDIENCE` + `GMAIL_PUBSUB_SERVICE_ACCOUNT` (or `GMAIL_PUBSUB_VERIFICATION_TOKEN`). |
| Push returns `403` | The OIDC audience/service account, verification token, or subscription name does not match the `.env` values. Check them after a tunnel URL change. |
| Push works but nothing is answered | Auto-replies are off by default. Turn on **Reply automatically to new emails**. Automated and self-sent mail is never answered. |

## Project Structure

```
├── agent/                  # Agentic chat layer
│   ├── controller.py       # Multi-turn tool-use loop
│   ├── llm_client.py       # Groq API client
│   ├── tools.py            # Tool implementations (search, send_email_reply)
│   ├── tool_executer.py    # Tool registry and execution
│   ├── state.py            # Conversation state management
│   └── system_prompt.py    # System prompt composition
├── database/               # Data persistence
│   ├── models.py           # SQLAlchemy ORM models
│   ├── postgresql.py       # PostgreSQL CRUD operations
│   └── base_database.py    # Database ABC
├── loaders/                # Platform data loaders
│   ├── gmail_loader.py     # Google API client
│   ├── slack_loader.py     # Slack Web API via httpx
│   ├── discord_loader.py   # Discord REST API via httpx
│   ├── telegram_loader.py  # Telethon (MTProto)
│   └── pdf_loader.py       # PyMuPDF extraction
├── rag/                    # RAG pipeline
│   ├── pipeline.py         # Orchestrates clean → chunk → embed → store
│   ├── chunker.py          # LangChain text splitter
│   ├── embedder.py         # SentenceTransformer embeddings
│   ├── preprocessing/      # Platform-specific cleaners
│   └── vector_store/       # ChromaDB vector store
├── routers/                # FastAPI route handlers (incl. gmail_push.py)
├── schemas/                # Request/response models
├── services/               # Auth services per platform, Gmail API, push sync, auto-responder
├── static/                 # Frontend (single-page app)
├── tests/                  # pytest suite (SQLite + mocked Gmail/LLM)
├── config.py               # Settings via pydantic-settings
├── initializer.py          # Shared singletons
├── main.py                 # App entrypoint
├── models.py               # Pydantic data models
├── .env.example            # Environment variable template (no secrets)
├── requirements.txt        # Python dependencies
└── requirements-dev.txt    # Test dependencies (pytest)
```

## Tech Stack

- **Framework**: FastAPI + Uvicorn
- **Database**: PostgreSQL + SQLAlchemy
- **Vector Store**: ChromaDB
- **Embeddings**: SentenceTransformers (`all-mpnet-base-v2`)
- **LLM**: Groq API
- **Auth**: JWT (PyJWT) + Argon2 password hashing (pwdlib)
