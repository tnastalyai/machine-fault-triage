import sys
from pathlib import Path

import pytest

from machine_fault_triage import CachedEmbedder
from machine_fault_triage.cli import main
from machine_fault_triage.evals import Metric, Pricing, RunReport
from machine_fault_triage.testing import FakeEmbedder
from tests.evals.factories import CASE_TOML, DOCUMENT_TOML, write_dataset
from tests.pdf_files import write_pdf

CORRECT = "tests.evals.factories:correct_model"
ABSTAINING = "tests.evals.factories:abstaining_model"
EMBEDDER = "tests.evals.factories:configured_embedder"


def run(dataset: Path, model: str, out: Path) -> int:
    return main(["eval", "run", "--dataset", str(dataset), "--model", model, "--out", str(out)])


def test_eval_run_writes_a_report_and_prints_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run.json"

    exit_code = run(write_dataset(tmp_path / "tiny"), CORRECT, out)

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    printed = capsys.readouterr().out
    assert exit_code == 0
    assert (report.dataset, report.model, len(report.cases)) == ("tiny", CORRECT, 1)
    assert f"dataset tiny  model {CORRECT}" in printed
    assert "top_cause                      1    1.00       -           1/1" in printed


def test_eval_compare_prints_the_difference_between_two_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path / "tiny")
    run(dataset, ABSTAINING, tmp_path / "old.json")
    run(dataset, CORRECT, tmp_path / "new.json")
    capsys.readouterr()

    exit_code = main(["eval", "compare", str(tmp_path / "old.json"), str(tmp_path / "new.json")])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "top_cause                      1    0.00    1.00   +1.00       -" in printed
    assert "improved  top_cause: e7" in printed


def test_pages_option_limits_the_pages_sent_per_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run.json"
    dataset = write_dataset(tmp_path / "tiny")

    exit_code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--model",
            CORRECT,
            "--out",
            str(out),
            "--pages",
            "1",
        ]
    )

    assert exit_code == 0
    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.page_selection == "keyword, at most 1 pages"
    assert "pages: keyword, at most 1 pages" in capsys.readouterr().out


SECOND_PAGE = """
[[pages]]
number = 78
text = "7.4 Cleaning the strainer"
"""


def run_with(tmp_path: Path, *options: str) -> tuple[int, Path]:
    out = tmp_path / "run.json"
    dataset = write_dataset(tmp_path / "tiny", document=DOCUMENT_TOML + SECOND_PAGE)
    arguments = ["eval", "run", "--dataset", str(dataset), "--model", CORRECT, "--out", str(out)]
    return main([*arguments, *options]), out


def test_neighbours_and_embedder_options_shape_the_page_selection(tmp_path: Path) -> None:
    cache = tmp_path / "vectors.sqlite"

    exit_code, out = run_with(
        tmp_path,
        *("--pages", "1", "--neighbours", "1"),
        *("--embedder", EMBEDDER, "--embedder-option", "word=flow"),
        *("--embedding-cache", str(cache)),
    )

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report.page_selection == (
        f"hybrid with {EMBEDDER} word=flow, at most 1 pages, 1 neighbouring on each side"
    )
    assert report.cases[0].repeats[0].scores[Metric.EXPECTED_PAGE_SENT] == 1
    assert cache.is_file()


def test_hybrid_selection_works_without_a_cache(tmp_path: Path) -> None:
    exit_code, _ = run_with(tmp_path, "--pages", "1", "--embedder", EMBEDDER)

    assert exit_code == 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--neighbours", "1"), "need --pages"),
        (("--embedder", EMBEDDER), "need --pages"),
        (("--pages", "1", "--embedding-cache", "vectors.sqlite"), "needs --embedder"),
        (
            ("--pages", "1", "--embedder", "tests.evals.factories:missing"),
            "cannot load the embedder",
        ),
        (("--pages", "0"), "limit must be at least 1"),
        (("--chunk-size", "500"), "need --pages"),
        (("--pages", "1", "--chunk-overlap", "50"), "needs --chunk-size"),
        (("--pages", "1", "--chunk-size", "50", "--chunk-overlap", "50"), "chunk_size must be"),
        (("--pages", "1", "--neighbours", "-1"), "neighbours must not be negative"),
    ],
)
def test_page_options_that_do_not_fit_together_are_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: tuple[str, ...], message: str
) -> None:
    exit_code, _ = run_with(tmp_path, *options)

    assert exit_code == 2
    assert message in capsys.readouterr().err


def test_prompt_option_runs_an_older_prompt_version(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    dataset = write_dataset(tmp_path / "tiny")

    exit_code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--model",
            CORRECT,
            "--out",
            str(out),
            "--prompt",
            "1",
        ]
    )

    assert exit_code == 0
    assert RunReport.model_validate_json(out.read_text(encoding="utf-8")).prompt_version == 1


def test_unknown_prompt_version_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path / "tiny")

    exit_code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--model",
            CORRECT,
            "--prompt",
            "99",
            "--out",
            str(tmp_path / "run.json"),
        ]
    )

    assert exit_code == 2
    assert "there is no prompt version 99" in capsys.readouterr().err


def test_report_directory_is_created_when_missing(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "today" / "run.json"

    exit_code = run(write_dataset(tmp_path / "tiny"), CORRECT, out)

    assert exit_code == 0
    assert out.is_file()


def test_report_path_that_cannot_be_created_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "taken").write_text("a file, not a directory", encoding="utf-8")

    exit_code = run(write_dataset(tmp_path / "tiny"), CORRECT, tmp_path / "taken" / "run.json")

    assert exit_code == 2
    assert "cannot write a report" in capsys.readouterr().err


def test_price_is_recorded_in_the_report(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    arguments = ["--dataset", str(write_dataset(tmp_path / "tiny")), "--model", CORRECT]

    prices = ["--price", "2", "10", "--cached-price", "0.2", "--embedding-price", "0.02"]

    exit_code = main(["eval", "run", *arguments, *prices, "--out", str(out)])

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report.pricing == Pricing(input=2, output=10, cached_input=0.2, embedding=0.02)


def test_cached_and_embedding_prices_need_the_model_price(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--dataset", str(write_dataset(tmp_path / "tiny")), "--model", CORRECT]

    exit_code = main(
        ["eval", "run", *arguments, "--embedding-price", "0.02", "--out", str(tmp_path / "r.json")]
    )

    assert exit_code == 2
    assert "--cached-price and --embedding-price need --price" in capsys.readouterr().err


def test_negative_price_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--dataset", str(write_dataset(tmp_path / "tiny")), "--model", CORRECT]

    exit_code = main(
        ["eval", "run", *arguments, "--price", "-1", "10", "--out", str(tmp_path / "run.json")]
    )

    assert exit_code == 2
    assert "a price cannot be negative" in capsys.readouterr().err


def test_label_overrides_the_model_name_in_the_report(tmp_path: Path) -> None:
    out = tmp_path / "run.json"

    main(
        [
            "eval",
            "run",
            "--dataset",
            str(write_dataset(tmp_path / "tiny")),
            "--model",
            CORRECT,
            "--label",
            "scripted",
            "--out",
            str(out),
        ]
    )

    assert RunReport.model_validate_json(out.read_text(encoding="utf-8")).model == "scripted"


@pytest.mark.parametrize(
    "model",
    ["tests.evals.factories:missing", "no_such_module:factory", "", "tests.evals.factories"],
)
def test_unloadable_model_is_reported_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model: str
) -> None:
    exit_code = run(write_dataset(tmp_path / "tiny"), model, tmp_path / "run.json")

    assert exit_code == 2
    assert "cannot load the model from" in capsys.readouterr().err


def test_broken_dataset_is_reported_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path / "tiny", case=CASE_TOML.replace("page = 77", "page = 1"))

    exit_code = run(dataset, CORRECT, tmp_path / "run.json")

    assert exit_code == 2
    assert "does not supply" in capsys.readouterr().err


def test_unreadable_report_is_reported_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "broken.json").write_text("{}", encoding="utf-8")

    exit_code = main(
        ["eval", "compare", str(tmp_path / "broken.json"), str(tmp_path / "missing.json")]
    )

    assert exit_code == 2
    assert "cannot read a report" in capsys.readouterr().err


def test_options_are_passed_to_the_model_callable(tmp_path: Path) -> None:
    out = tmp_path / "run.json"

    exit_code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(write_dataset(tmp_path / "tiny")),
            "--model",
            "tests.evals.factories:configured_model",
            "--option",
            "answer=correct",
            "--out",
            str(out),
        ]
    )

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report.cases[0].repeats[0].scores[Metric.TOP_CAUSE] == 1


@pytest.mark.parametrize("option", ["answer", "=correct", "unknown=1"])
def test_option_the_model_callable_cannot_take_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], option: str
) -> None:
    exit_code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(write_dataset(tmp_path / "tiny")),
            "--model",
            "tests.evals.factories:configured_model",
            "--option",
            option,
            "--out",
            str(tmp_path / "run.json"),
        ]
    )

    assert exit_code == 2
    assert "cannot load the model from" in capsys.readouterr().err


def sync(tmp_path: Path, *options: str) -> int:
    folder, registry = tmp_path / "manuals", tmp_path / "registry.sqlite"
    return main(["library", "sync", "--folder", str(folder), "--registry", str(registry), *options])


def test_library_sync_lists_what_changed_and_deletes_stale_vectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "vectors.sqlite"
    path = tmp_path / "manuals/acme/pump-1/service.pdf"
    path.parent.mkdir(parents=True)
    write_pdf(path, [["Old page."]])
    sync(tmp_path, "--embedding-cache", str(cache))
    CachedEmbedder(FakeEmbedder({"page": [1.0]}), cache, model="m1").embed_pages(["Old page."])
    write_pdf(path, [["New page."]])

    exit_code = sync(tmp_path, "--embedding-cache", str(cache))

    assert exit_code == 0
    assert capsys.readouterr().out == (
        "added     acme/pump-1/service\n"
        "note      acme/pump-1/service: 9 characters per page on average;"
        " the text may be incomplete\n"
        "changed   acme/pump-1/service\n"
        "note      acme/pump-1/service: 9 characters per page on average;"
        " the text may be incomplete\n"
        "1 stored vectors deleted\n"
    )


def test_library_sync_of_an_empty_folder_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "manuals").mkdir()

    assert sync(tmp_path) == 0
    assert capsys.readouterr().out == "the folder holds no manuals\n"


def test_library_sync_of_a_missing_folder_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sync(tmp_path) == 2
    assert "cannot be read" in capsys.readouterr().err


def test_library_sync_without_the_pdf_extra_says_what_to_install(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "machine_fault_triage.folder", None)

    assert sync(tmp_path) == 2
    assert "needs the pdf extra" in capsys.readouterr().err


def test_chunk_options_rank_pieces_and_record_retrieval_metrics(tmp_path: Path) -> None:
    exit_code, out = run_with(tmp_path, "--pages", "1", "--chunk-size", "40")

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    scores = report.cases[0].repeats[0].scores
    assert exit_code == 0
    assert report.page_selection == (
        "keyword, at most 1 pages, ranked by recursive pieces of 40 characters"
    )
    assert scores[Metric.RECIPROCAL_RANK] == 1
    assert scores[Metric.RETRIEVAL_PRECISION] == 1


def test_fixed_overlapping_pieces_can_be_chosen(tmp_path: Path) -> None:
    exit_code, out = run_with(
        tmp_path, "--pages", "1", "--chunk-size", "40", "--chunk-overlap", "10"
    )

    report = RunReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "fixed pieces of 40 characters, 10 overlapping" in str(report.page_selection)


def test_chunk_size_without_the_chunking_extra_says_what_to_install(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "machine_fault_triage.adapters.chonkie", None)

    exit_code, _ = run_with(tmp_path, "--pages", "1", "--chunk-size", "40")

    assert exit_code == 2
    assert "needs the chunking extra" in capsys.readouterr().err
