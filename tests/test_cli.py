from pathlib import Path

import pytest

from machine_fault_triage.cli import main
from machine_fault_triage.evals import Metric, RunReport
from tests.evals.factories import CASE_TOML, write_dataset

CORRECT = "tests.evals.factories:correct_model"
ABSTAINING = "tests.evals.factories:abstaining_model"


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
    assert "cannot load a model from" in capsys.readouterr().err


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
    assert "cannot load a model from" in capsys.readouterr().err
