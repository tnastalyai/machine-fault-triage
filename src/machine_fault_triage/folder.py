"""PDF manuals kept in a folder, with a registry of what was read. Needs the ``pdf`` extra."""

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from machine_fault_triage import pdf
from machine_fault_triage.embeddings import page_fingerprint
from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.inputs import Document

_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    tenant TEXT NOT NULL,
    device TEXT NOT NULL,
    name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    extraction INTEGER NOT NULL,
    synced_at TEXT NOT NULL,
    document TEXT NOT NULL,
    PRIMARY KEY (tenant, device, name)
)
"""

_SOURCE = "CREATE TABLE IF NOT EXISTS source (folder TEXT NOT NULL)"

_LONG_PAGE = 6000
_THIN_TEXT = 200

type _Key = tuple[str, str, str]


class SyncReport(BaseModel):
    """What one ``FolderLibrary.sync`` found, each file as ``tenant/device/name``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    """Files that could not be read. A version read earlier stays in use."""
    forgotten_pages: tuple[str, ...] = ()
    """Fingerprints of page texts that no document holds any more."""
    notes: tuple[str, ...] = ()
    """Why a file failed, and what may limit the answers from a file that was read."""


class FolderLibrary:
    """Manuals of many devices, read from ``folder/<tenant>/<device>/<name>.pdf``.

    ``sync`` reads new and changed files into the registry, a local SQLite
    file, and drops files that are gone. ``documents_for`` answers from the
    registry alone, so a triage run sees the state of the last sync and never
    parses a PDF. The file name without ``.pdf`` is the document id and title.

    The tenant and the device come from the folder names at sync time and are
    looked up by equality, so no id can reach outside its folder. Symbolic
    links are not followed. A folder cannot say that one manual supersedes
    another; remove the old file instead. The registry holds the text of the
    manuals: keep it where the manuals may be kept.
    """

    def __init__(self, folder: Path, registry: Path) -> None:
        """Choose the folder of manuals and the registry file, created on first use."""
        self._folder = folder
        self._registry = registry

    def sync(self) -> SyncReport:
        """Bring the registry in line with the folder, in one transaction.

        A file is read again when its content changed or when ``load_pdf``
        has changed how it extracts text. A file that cannot be read is
        reported as failed and does not stop the others.

        Raises:
            DocumentLoadError: If the folder is missing or unreadable, if it
                holds no manuals while the registry does, or if the registry
                was filled from another folder. Nothing is removed then, so an
                unmounted disk or a mistyped path does not empty the registry.
                To start over on purpose, delete the registry file.
        """
        files = dict(self._files())
        outcome: dict[str, list[str]] = {
            name: [] for name in ("added", "changed", "removed", "unchanged", "failed", "notes")
        }
        with closing(sqlite3.connect(self._registry)) as connection, connection:
            connection.execute(_TABLE)
            stored = {
                (tenant, device, name): (file_hash, extraction)
                for tenant, device, name, file_hash, extraction in connection.execute(
                    "SELECT tenant, device, name, file_hash, extraction FROM documents"
                )
            }
            self._check_source(connection, has_files=bool(files), has_stored=bool(stored))
            before = _page_fingerprints(connection)
            for key, path in files.items():
                try:
                    content = path.read_bytes()
                except OSError as error:
                    outcome["failed"].append("/".join(key))
                    outcome["notes"].append(f"{'/'.join(key)}: {error.strerror}")
                    continue
                state = (hashlib.sha256(content).hexdigest(), pdf.EXTRACTION_VERSION)
                if stored.get(key) == state:
                    outcome["unchanged"].append("/".join(key))
                    continue
                try:
                    document = pdf.load_pdf(path, document_id=key[2], title=key[2])
                except DocumentLoadError as error:
                    outcome["failed"].append("/".join(key))
                    outcome["notes"].append(f"{'/'.join(key)}: {error}")
                    continue
                outcome["notes"].extend(_limits("/".join(key), document))
                outcome["changed" if key in stored else "added"].append("/".join(key))
                connection.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (*key, *state, datetime.now(UTC).isoformat(), document.model_dump_json()),
                )
            for key in sorted(stored.keys() - files.keys()):
                outcome["removed"].append("/".join(key))
                connection.execute(
                    "DELETE FROM documents WHERE tenant = ? AND device = ? AND name = ?", key
                )
            forgotten = sorted(before - _page_fingerprints(connection))
        return SyncReport(**outcome, forgotten_pages=forgotten)

    def documents_for(self, *, tenant_id: str, device_id: str) -> Sequence[Document]:
        """Return the manuals of one device of one tenant, as of the last sync."""
        with closing(sqlite3.connect(self._registry)) as connection, connection:
            connection.execute(_TABLE)
            rows = connection.execute(
                "SELECT document FROM documents WHERE tenant = ? AND device = ? ORDER BY name",
                (tenant_id, device_id),
            ).fetchall()
        return [Document.model_validate_json(row[0]) for row in rows]

    def _check_source(
        self, connection: sqlite3.Connection, *, has_files: bool, has_stored: bool
    ) -> None:
        folder = str(self._folder.resolve())
        connection.execute(_SOURCE)
        row = connection.execute("SELECT folder FROM source").fetchone()
        if row is None:
            connection.execute("INSERT INTO source VALUES (?)", (folder,))
        elif row[0] != folder:
            message = f"the registry {self._registry} was filled from {row[0]}, not from {folder}"
            raise DocumentLoadError(message)
        if has_stored and not has_files:
            message = f"the folder {self._folder} holds no manuals, but the registry does"
            raise DocumentLoadError(message)

    def _files(self) -> Iterator[tuple[_Key, Path]]:
        try:
            tenants = _folders(self._folder)
        except OSError as error:
            message = f"the folder {self._folder} cannot be read: {error}"
            raise DocumentLoadError(message) from error
        for tenant in tenants:
            for device in _folders(tenant):
                for path in sorted(device.glob("*.pdf")):
                    if path.is_file() and not path.is_symlink():
                        yield (tenant.name, device.name, path.stem), path


def _limits(name: str, document: Document) -> list[str]:
    notes = []
    without_text = document.pages[-1].number - len(document.pages)
    if without_text:
        notes.append(f"{name}: {without_text} pages have no text and cannot be searched or quoted")
    longest = max(document.pages, key=lambda page: len(page.text))
    if len(longest.text) > _LONG_PAGE:
        notes.append(
            f"{name}: page {longest.number} has {len(longest.text)} characters;"
            " ranking by smaller pieces may find its content better"
        )
    mean = sum(len(page.text) for page in document.pages) / len(document.pages)
    if mean < _THIN_TEXT:
        notes.append(
            f"{name}: {mean:.0f} characters per page on average; the text may be incomplete"
        )
    return notes


def _folders(parent: Path) -> list[Path]:
    return sorted(path for path in parent.iterdir() if path.is_dir() and not path.is_symlink())


def _page_fingerprints(connection: sqlite3.Connection) -> set[str]:
    return {
        page_fingerprint(page.text)
        for (document,) in connection.execute("SELECT document FROM documents")
        for page in Document.model_validate_json(document).pages
    }
