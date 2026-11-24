"""Command line interface: ``machine-fault-triage eval run`` and ``eval compare``."""

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from machine_fault_triage.errors import InvalidRequestError, TriageError
from machine_fault_triage.evals.cases import load_dataset
from machine_fault_triage.evals.compare import compare_runs, render_comparison
from machine_fault_triage.evals.report import Pricing, RunReport, render_summary
from machine_fault_triage.evals.runner import run_eval
from machine_fault_triage.ports import StructuredModel

_USAGE_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return its exit code."""
    arguments = _parser().parse_args(argv)
    try:
        output = _compare(arguments) if arguments.command == "compare" else _run(arguments)
    except TriageError as error:
        sys.stderr.write(f"error: {error}\n")
        return _USAGE_ERROR
    sys.stdout.write(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="machine-fault-triage")
    evaluation = parser.add_subparsers(required=True).add_parser(
        "eval", help="measure a model on a dataset"
    )
    commands = evaluation.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a dataset against a model and write a report")
    run.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    run.add_argument(
        "--model",
        required=True,
        help="import path of a callable that returns the model, as module:callable",
    )
    run.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument passed to the model callable; repeatable",
    )
    run.add_argument("--label", help="model name recorded in the report")
    run.add_argument("--repeats", type=int, default=1, help="runs per case")
    run.add_argument(
        "--price",
        type=float,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="model price in US dollars per million input and output tokens",
    )
    run.add_argument("--out", type=Path, required=True, help="report file to write")
    compare = commands.add_parser("compare", help="compare two reports case by case")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    return parser


def _run(arguments: argparse.Namespace) -> str:
    try:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        message = f"cannot write a report to {arguments.out}: {error}"
        raise InvalidRequestError(message) from error
    report = run_eval(
        load_dataset(arguments.dataset),
        _load_model(arguments.model, arguments.option),
        model_label=arguments.label or arguments.model,
        repeats=arguments.repeats,
        pricing=_pricing(arguments.price),
    )
    arguments.out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return render_summary(report)


def _pricing(price: Sequence[float] | None) -> Pricing | None:
    if price is None:
        return None
    try:
        return Pricing(input=price[0], output=price[1])
    except ValidationError as error:
        message = f"a price cannot be negative: {error}"
        raise InvalidRequestError(message) from error


def _compare(arguments: argparse.Namespace) -> str:
    return render_comparison(
        compare_runs(_read_report(arguments.baseline), _read_report(arguments.candidate))
    )


def _load_model(path: str, options: Sequence[str]) -> StructuredModel:
    module_name, _, attribute = path.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        model: StructuredModel = factory(**_keyword_arguments(options))
    except (ImportError, AttributeError, ValueError, TypeError) as error:
        message = f"cannot load a model from {path!r}: {error}"
        raise InvalidRequestError(message) from error
    return model


def _keyword_arguments(options: Sequence[str]) -> dict[str, str]:
    pairs = [option.partition("=") for option in options]
    if not all(key and separator for key, separator, _ in pairs):
        message = "an option must look like KEY=VALUE"
        raise ValueError(message)
    return {key: value for key, _, value in pairs}


def _read_report(path: Path) -> RunReport:
    try:
        return RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        message = f"cannot read a report from {path}: {error}"
        raise InvalidRequestError(message) from error
