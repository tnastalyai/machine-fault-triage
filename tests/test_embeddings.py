from pathlib import Path

from machine_fault_triage import CachedEmbedder
from machine_fault_triage.testing import FakeEmbedder

MEANINGS = {"pump": [1.0, 0.0], "sensor": [0.0, 1.0]}


def test_each_page_is_embedded_once_across_runs(tmp_path: Path) -> None:
    first = FakeEmbedder(MEANINGS)
    second = FakeEmbedder(MEANINGS)
    path = tmp_path / "vectors.sqlite"

    CachedEmbedder(first, path, model="m1").embed_pages(["pump page", "sensor page"])
    vectors = CachedEmbedder(second, path, model="m1").embed_pages(
        ["sensor page", "pump page", "another pump page"]
    )

    assert vectors.vectors == [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    assert first.page_calls == [["pump page", "sensor page"]]
    assert second.page_calls == [["another pump page"]]


def test_same_page_twice_in_one_call_is_embedded_once(tmp_path: Path) -> None:
    embedder = FakeEmbedder(MEANINGS)

    vectors = CachedEmbedder(embedder, tmp_path / "v.sqlite", model="m1").embed_pages(
        ["pump page", "pump page"]
    )

    assert vectors.vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.page_calls == [["pump page"]]


def test_fully_stored_pages_need_no_call_at_all(tmp_path: Path) -> None:
    path = tmp_path / "vectors.sqlite"
    CachedEmbedder(FakeEmbedder(MEANINGS), path, model="m1").embed_pages(["pump page"])
    idle = FakeEmbedder(MEANINGS)

    CachedEmbedder(idle, path, model="m1").embed_pages(["pump page"])

    assert idle.page_calls == []


def test_another_model_does_not_reuse_stored_vectors(tmp_path: Path) -> None:
    path = tmp_path / "vectors.sqlite"
    CachedEmbedder(FakeEmbedder(MEANINGS), path, model="m1").embed_pages(["pump page"])
    other = FakeEmbedder(MEANINGS)

    CachedEmbedder(other, path, model="m2").embed_pages(["pump page"])

    assert other.page_calls == [["pump page"]]


def test_fault_reports_are_passed_through_and_never_stored(tmp_path: Path) -> None:
    path = tmp_path / "vectors.sqlite"
    embedder = FakeEmbedder(MEANINGS)
    cached = CachedEmbedder(embedder, path, model="m1")

    assert cached.embed_query("the pump hums").vectors == [[1.0, 0.0]]
    assert cached.embed_query("the pump hums").vectors == [[1.0, 0.0]]
    assert embedder.queries == ["the pump hums", "the pump hums"]
    assert not path.exists()


def test_only_newly_embedded_pages_and_reports_cost_tokens(tmp_path: Path) -> None:
    path = tmp_path / "vectors.sqlite"
    cached = CachedEmbedder(FakeEmbedder(MEANINGS, tokens_per_text=7), path, model="m1")

    first = cached.embed_pages(["pump page", "sensor page"])
    second = cached.embed_pages(["pump page", "sensor page", "another pump page"])
    third = cached.embed_pages(["pump page"])

    assert (first.tokens, second.tokens, third.tokens) == (14, 7, 0)
    assert cached.embed_query("the pump hums").tokens == 7
