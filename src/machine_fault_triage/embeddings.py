"""Keeping page vectors between runs, so that each page is embedded once."""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path

from machine_fault_triage.ports import Embedder, Embeddings

_TABLE = """
CREATE TABLE IF NOT EXISTS page_vectors (
    model TEXT NOT NULL,
    page TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (model, page)
)
"""


def page_fingerprint(text: str) -> str:
    """Return the hash under which the vector of a page text is stored."""
    return hashlib.sha256(text.encode()).hexdigest()


def forget_pages(path: Path, fingerprints: Iterable[str]) -> int:
    """Delete the stored vectors of the given pages, for every model.

    Returns the number of vectors deleted. The rows are deleted and the file
    is compacted, so the vectors do not stay in free space of the file.
    """
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(_TABLE)
            deleted = sum(
                connection.execute(
                    "DELETE FROM page_vectors WHERE page = ?", (fingerprint,)
                ).rowcount
                for fingerprint in fingerprints
            )
        connection.execute("VACUUM")
    return deleted


class CachedEmbedder:
    """An embedder that stores the vectors of pages in a local SQLite file.

    A vector is stored under the embedding model and a hash of the page text,
    so a changed page or another model is embedded again. Fault reports are
    passed through and never stored.
    """

    def __init__(self, embedder: Embedder, path: Path, *, model: str) -> None:
        """Wrap ``embedder`` and keep its page vectors in the file at ``path``.

        ``model`` names the embedding model. Vectors of different models are
        not comparable, so it is part of every stored key.
        """
        self._embedder = embedder
        self._path = path
        self._model = model

    def embed_pages(self, texts: Sequence[str]) -> Embeddings:
        """Return one vector per page, calling the wrapped embedder only for new pages.

        The tokens reported are those of the new pages; stored pages cost none.
        """
        pages = [page_fingerprint(text) for text in texts]
        tokens = 0
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(_TABLE)
            known: dict[str, list[float]] = {}
            for page in dict.fromkeys(pages):
                row = connection.execute(
                    "SELECT vector FROM page_vectors WHERE model = ? AND page = ?",
                    (self._model, page),
                ).fetchone()
                if row is not None:
                    known[page] = json.loads(row[0])
            missing = {
                page: text for page, text in zip(pages, texts, strict=True) if page not in known
            }
            if missing:
                embedded = self._embedder.embed_pages(list(missing.values()))
                tokens = embedded.tokens
                for page, vector in zip(missing, embedded.vectors, strict=True):
                    known[page] = list(vector)
                    connection.execute(
                        "INSERT INTO page_vectors (model, page, vector) VALUES (?, ?, ?)",
                        (self._model, page, json.dumps(known[page])),
                    )
        return Embeddings([known[page] for page in pages], tokens)

    def embed_query(self, text: str) -> Embeddings:
        """Return the vector of a fault report without storing it."""
        return self._embedder.embed_query(text)
