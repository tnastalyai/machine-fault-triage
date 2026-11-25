import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from machine_fault_triage import CachedEmbedder, pdf
from machine_fault_triage.embeddings import forget_pages, page_fingerprint
from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.folder import FolderLibrary
from machine_fault_triage.models import PageSource
from machine_fault_triage.testing import FakeEmbedder
from tests.pdf_files import scripted_engine, write_pdf


def manual(folder: Path, name: str, pages: list[list[str]]) -> Path:
    path = folder / f"{name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_pdf(path, pages)


def library(tmp_path: Path) -> FolderLibrary:
    return FolderLibrary(tmp_path / "manuals", tmp_path / "registry.sqlite")


def test_synced_manual_is_served_to_its_device(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manuals = library(tmp_path)

    report = manuals.sync()

    (document,) = manuals.documents_for(tenant_id="acme", device_id="pump-1")
    assert report.added == ("acme/pump-1/service",)
    assert (document.id, document.title) == ("service", "service")
    assert document.pages[0].text == "E7: flow sensor fault."


def test_one_tenant_never_receives_the_manual_of_another(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["Acme only."]])
    manual(tmp_path / "manuals/globex/pump-1", "service", [["Globex only."]])
    manuals = library(tmp_path)
    manuals.sync()

    (document,) = manuals.documents_for(tenant_id="globex", device_id="pump-1")

    assert document.pages[0].text == "Globex only."
    assert manuals.documents_for(tenant_id="acme/../globex", device_id="pump-1") == []
    assert manuals.documents_for(tenant_id="acme", device_id="pump-2") == []


def test_second_sync_reads_nothing_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manuals = library(tmp_path)
    manuals.sync()
    monkeypatch.setattr(pdf, "load_pdf", None)

    report = manuals.sync()

    assert report.unchanged == ("acme/pump-1/service",)
    assert report.added == report.changed == report.forgotten_pages == ()


def test_changed_file_replaces_the_stored_manual(tmp_path: Path) -> None:
    folder = tmp_path / "manuals/acme/pump-1"
    manual(folder, "service", [["E7: flow sensor fault."], ["Kept page."]])
    manuals = library(tmp_path)
    manuals.sync()
    manual(folder, "service", [["E7: flow switch fault."], ["Kept page."]])

    report = manuals.sync()

    (document,) = manuals.documents_for(tenant_id="acme", device_id="pump-1")
    assert report.changed == ("acme/pump-1/service",)
    assert document.pages[0].text == "E7: flow switch fault."
    assert report.forgotten_pages == (page_fingerprint("E7: flow sensor fault."),)


def test_removed_file_is_dropped_but_a_page_still_held_elsewhere_is_kept(tmp_path: Path) -> None:
    shared = ["Shared safety page."]
    old = manual(tmp_path / "manuals/acme/pump-1", "old", [shared, ["Only in the old one."]])
    manual(tmp_path / "manuals/acme/pump-1", "new", [shared])
    manuals = library(tmp_path)
    manuals.sync()
    old.unlink()

    report = manuals.sync()

    assert report.removed == ("acme/pump-1/old",)
    assert report.forgotten_pages == (page_fingerprint("Only in the old one."),)
    assert [d.id for d in manuals.documents_for(tenant_id="acme", device_id="pump-1")] == ["new"]


def test_unreadable_file_is_reported_and_the_earlier_version_stays(tmp_path: Path) -> None:
    folder = tmp_path / "manuals/acme/pump-1"
    path = manual(folder, "service", [["E7: flow sensor fault."]])
    manual(folder, "wiring", [["Terminal X3."]])
    manuals = library(tmp_path)
    manuals.sync()
    path.write_bytes(b"not a pdf")

    report = manuals.sync()

    assert report.failed == ("acme/pump-1/service",)
    assert report.unchanged == ("acme/pump-1/wiring",)
    assert "acme/pump-1/service: " in report.notes[0]
    assert "cannot be read as a PDF" in report.notes[0]
    documents = manuals.documents_for(tenant_id="acme", device_id="pump-1")
    assert documents[0].pages[0].text == "E7: flow sensor fault."


def test_file_without_read_permission_is_reported_and_does_not_stop_the_sync(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "manuals/acme/pump-1"
    locked = manual(folder, "locked", [["Secret."]])
    manual(folder, "wiring", [["Terminal X3."]])
    locked.chmod(0)

    report = library(tmp_path).sync()

    assert report.failed == ("acme/pump-1/locked",)
    assert report.notes[0] == "acme/pump-1/locked: Permission denied"
    assert report.added == ("acme/pump-1/wiring",)


def test_missing_folder_is_an_error_and_removes_nothing(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    library(tmp_path).sync()
    unmounted = FolderLibrary(tmp_path / "unmounted", tmp_path / "unmounted.sqlite")

    with pytest.raises(DocumentLoadError, match="cannot be read"):
        unmounted.sync()

    assert unmounted.documents_for(tenant_id="acme", device_id="pump-1") == []


def test_emptied_folder_is_an_error_and_removes_nothing(tmp_path: Path) -> None:
    path = manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manuals = library(tmp_path)
    manuals.sync()
    path.unlink()

    with pytest.raises(DocumentLoadError, match="holds no manuals, but the registry does"):
        manuals.sync()

    assert len(manuals.documents_for(tenant_id="acme", device_id="pump-1")) == 1


def test_registry_filled_from_one_folder_refuses_another(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manual(tmp_path / "other/globex/pump-9", "wiring", [["Terminal X3."]])
    library(tmp_path).sync()
    other = FolderLibrary(tmp_path / "other", tmp_path / "registry.sqlite")

    with pytest.raises(DocumentLoadError, match="was filled from"):
        other.sync()

    assert len(library(tmp_path).documents_for(tenant_id="acme", device_id="pump-1")) == 1


def test_symbolic_links_and_stray_files_are_ignored(tmp_path: Path) -> None:
    real = manual(tmp_path / "manuals/acme/pump-1", "service", [["Acme only."]])
    globex = tmp_path / "manuals/globex/pump-1"
    globex.mkdir(parents=True)
    (globex / "borrowed.pdf").symlink_to(real)
    (tmp_path / "manuals/linked").symlink_to(tmp_path / "manuals/acme")
    (tmp_path / "manuals/notes.txt").write_text("stray", encoding="utf-8")
    (tmp_path / "manuals/acme/pump-1/folder.pdf").mkdir()

    report = library(tmp_path).sync()

    assert report.added == ("acme/pump-1/service",)


def test_manual_is_read_again_when_text_extraction_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manuals = library(tmp_path)
    manuals.sync()
    monkeypatch.setattr(pdf, "EXTRACTION_VERSION", pdf.EXTRACTION_VERSION + 1)

    assert manuals.sync().changed == ("acme/pump-1/service",)


def test_vectors_of_forgotten_pages_are_deleted_for_every_model(tmp_path: Path) -> None:
    cache = tmp_path / "vectors.sqlite"
    for model in ("m1", "m2"):
        CachedEmbedder(FakeEmbedder({"page": [1.0]}), cache, model=model).embed_pages(
            ["old page", "kept page"]
        )

    deleted = forget_pages(cache, [page_fingerprint("old page")])

    again = FakeEmbedder({"page": [1.0]})
    CachedEmbedder(again, cache, model="m1").embed_pages(["old page", "kept page"])
    assert deleted == 2
    assert again.page_calls == [["old page"]]


def test_sync_notes_what_may_limit_the_answers_from_a_manual(tmp_path: Path) -> None:
    folder = tmp_path / "manuals/acme/pump-1"
    manual(
        folder,
        "gaps",
        [["Cover page with enough text to count as a normal page." * 5], [], ["Z" * 250]],
    )
    manual(folder, "long", [["word " * 15] * 90])
    manual(folder, "thin", [["E7"], ["A12"]])
    manual(folder, "scan", [[], []])

    report = library(tmp_path).sync()

    assert report.failed == ("acme/pump-1/scan",)
    gaps, long, scan, thin = report.notes
    assert gaps == "acme/pump-1/gaps: 1 pages have no text and cannot be searched or quoted"
    assert long == (
        "acme/pump-1/long: page 1 has 6838 characters;"
        " ranking by smaller pieces may find its content better"
    )
    assert scan.startswith("acme/pump-1/scan: ")
    assert scan.endswith("scanned documents need character recognition first")
    assert thin == "acme/pump-1/thin: 2 characters per page on average; the text may be incomplete"


def test_scanned_manual_is_recognised_stored_and_noted(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "scan", [[], ["Printed cover page."]])
    manuals = FolderLibrary(
        tmp_path / "manuals", tmp_path / "registry.sqlite", ocr=scripted_engine("A12: flow switch.")
    )

    report = manuals.sync()

    (document,) = manuals.documents_for(tenant_id="acme", device_id="pump-1")
    assert report.added == ("acme/pump-1/scan",)
    assert [(page.text, page.source) for page in document.pages] == [
        ("A12: flow switch.", PageSource.OCR),
        ("Printed cover page.", PageSource.TEXT_LAYER),
    ]
    assert (
        "acme/pump-1/scan: 1 pages were recognised from images;"
        " a misread character there cannot be detected"
    ) in report.notes


def test_manual_is_read_again_only_when_the_recognition_engine_changes(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."], []])
    folder, registry = tmp_path / "manuals", tmp_path / "registry.sqlite"
    FolderLibrary(folder, registry).sync()

    added = FolderLibrary(folder, registry, ocr=scripted_engine(name="engine 1")).sync()
    same = FolderLibrary(folder, registry, ocr=scripted_engine(name="engine 1")).sync()
    other = FolderLibrary(folder, registry, ocr=scripted_engine(name="engine 2")).sync()
    removed = FolderLibrary(folder, registry).sync()

    assert added.changed == other.changed == removed.changed == ("acme/pump-1/service",)
    assert same.unchanged == ("acme/pump-1/service",)
    (document,) = FolderLibrary(folder, registry).documents_for(
        tenant_id="acme", device_id="pump-1"
    )
    assert [page.number for page in document.pages] == [1]


def test_registry_written_before_recognition_existed_still_opens(tmp_path: Path) -> None:
    manual(tmp_path / "manuals/acme/pump-1", "service", [["E7: flow sensor fault."]])
    manuals = library(tmp_path)
    manuals.sync()
    with closing(sqlite3.connect(tmp_path / "registry.sqlite")) as connection, connection:
        connection.execute("ALTER TABLE documents DROP COLUMN ocr")

    (document,) = manuals.documents_for(tenant_id="acme", device_id="pump-1")

    assert document.pages[0].text == "E7: flow sensor fault."
    assert manuals.sync().unchanged == ("acme/pump-1/service",)


def test_sync_notes_pages_whose_meaning_is_in_small_pictures(tmp_path: Path) -> None:
    text = ["Heating is active. Defrost is running. A fault is present."] * 5
    path = tmp_path / "manuals/acme/pump-1/controller.pdf"
    path.parent.mkdir(parents=True)
    write_pdf(path, [text, text, text], icons=[4, 5, 14])

    report = library(tmp_path).sync()

    (note,) = report.notes
    assert note == (
        "acme/pump-1/controller: many small pictures, such as icons, on pages 2, 3;"
        " what the pictures show is not in the text"
    )
