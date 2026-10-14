"""
Vector store backed by the open-source ChromaDB client.

    The pipeline is expected to produce already-embedded chunks
    (via :class:`~embedder.SentenceTransformerEmbedder`).
    This store only persists them and retrieves similar results —
    it does **not** re-embed documents.

    Accepts embedded chunks in the format returned by
    ``SentenceTransformerEmbedder.embed_many()``:

    .. code-block:: python

        {
            "chunk":  {"id": str, "text": str,
                       "chunk_index": int, "metadata": dict},
            "vector": [float, ...],
        }

    Parameters
    ----------
    embedder:
        A ``SentenceTransformerEmbedder`` instance used to embed
        query strings at search time.
    persist_directory:
        Directory on disk where ChromaDB stores its index.
    collection_name:
        Name of the ChromaDB collection to use.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import json

import chromadb

from rag.embedder import SentenceTransformerEmbedder
from .base_vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_where(
    source_type: str | None,
    owner_user_id: int | None,
    exclude_email_id: str | None = None,
) -> Dict[str, Any] | None:
    """Build the Chroma metadata filter for a search.

    Email chunks belong to one user's mailbox, so they are only returned
    for that user (and never when no user is known). Other sources keep
    their existing behaviour.
    """
    if source_type and source_type != "email":
        return {"source_type": source_type}

    if source_type == "email":
        if owner_user_id is None:
            # Unscoped email search would expose other users' mail.
            return {"$and": [{"source_type": "email"}, {"owner_user_id": -1}]}
        conditions: List[Dict[str, Any]] = [
            {"source_type": "email"},
            {"owner_user_id": owner_user_id},
        ]
        if exclude_email_id:
            conditions.append({"email_id": {"$ne": exclude_email_id}})
        return {"$and": conditions}

    # All sources: everything except other users' email.
    if owner_user_id is None:
        return {"source_type": {"$ne": "email"}}
    return {
        "$or": [
            {"source_type": {"$ne": "email"}},
            {"owner_user_id": owner_user_id},
        ]
    }


class ChromaVectorStore(VectorStore):
    """Vector store backed by ChromaDB with persistent storage."""

    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        persist_directory: str,
        collection_name: str,
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)

        self._embedder = embedder
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

        logger.info(
            "ChromaVectorStore ready — collection=%s, persist=%s",
            collection_name,
            persist_directory,
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Store pre-computed embedded chunks in ChromaDB.

        Expects the output format of
        ``SentenceTransformerEmbedder.embed_many()`` —
        each dict contains ``"chunk"`` (the original chunk dict) and
        ``"vector"`` (its pre-computed embedding).

        Embeddings are passed directly to ChromaDB and are **not**
        recomputed.
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks to add.")
            return

        documents = []
        metadata = []
        ids = []
        embeddings = []

        for item in embedded_chunks:
            chunk = item["chunk"]
            documents.append(chunk["text"])
            meta = dict(chunk.get("metadata", {}))
            meta["_id"] = chunk.get("id", "")
            # ChromaDB only accepts str, int, float, or bool metadata values.
            sanitized = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                else:
                    sanitized[k] = json.dumps(v)
            metadata.append(sanitized)
            ids.append(chunk.get("id", ""))
            embeddings.append(item["vector"])

        # Upsert keeps re-ingesting the same document idempotent: a retry
        # after a partial failure overwrites chunks instead of duplicating.
        self._collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids,
            embeddings=embeddings,
        )

        logger.info(
            "Added %d embedded chunks to collection '%s'",
            len(embedded_chunks),
            self._collection_name,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        source_type: str | None = None,
        owner_user_id: int | None = None,
        exclude_email_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-*k* chunks most similar to *query*.

        The query is embedded using the stored embedder; results are
        reconstructed into the original chunk dict schema.
        """
        query_vector = self._embedder.encode([query])[0]
        where = build_where(source_type, owner_user_id, exclude_email_id)
        results = self._collection.query(query_embeddings=[query_vector], n_results=k, where=where)

        if not results["ids"] or not results["ids"][0]:
            return []

        chunks = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            meta = dict(results["metadatas"][0][i]) if results["metadatas"] else {}
            chunk_id = meta.pop("_id", doc_id)
            chunks.append({
                "id": chunk_id,
                "text": text,
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": meta,
            })

        return chunks

    def has_email(self, email_id: str, owner_user_id: int | None = None) -> bool:
        """Return True if an email with the given ID already exists
        (in `owner_user_id`'s mailbox, when given)."""
        if not email_id:
             return False

        conditions: List[Dict[str, Any]] = [
            {"source_type": "email"},
            {"email_id": email_id},
        ]
        if owner_user_id is not None:
            conditions.append({"owner_user_id": owner_user_id})

        results = self._collection.get(
            where={"$and": conditions},
            limit=1,
        )

        return bool(results.get("ids"))

    def delete(self, ids: Optional[List[str]] = None) -> None:
        if ids is None:
            self._collection.delete()
            logger.info("Cleared entire collection '%s'", self._collection_name)
        else:
            self._collection.delete(ids=ids)
            logger.info(
                "Deleted %d documents from collection '%s'",
                len(ids),
                self._collection_name,
            )

    def persist(self) -> None:
        """
        ChromaDB's ``PersistentClient`` writes to disk automatically,
        but this method provides an explicit flush guarantee.
        """
        self._client.heartbeat()
        logger.info("Persisted collection '%s' to disk", self._collection_name)

    def count(self) -> int:
        return self._collection.count()
