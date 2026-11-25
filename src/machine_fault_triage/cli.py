"""Command line interface: ``machine-fault-triage eval run`` and ``eval compare``."""

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from machine_fault_triage.embeddings import CachedEmbedder, forget_pages
from machine_fault_triage.errors import InvalidRequestError, TriageError
from machine_fault_triage.evals.cases import load_dataset
from machine_fault_triage.evals.compare import compare_runs, render_comparison
from machine_fault_triage.evals.report import Pricing, RunReport, render_summary
from machine_fault_triage.evals.runner import run_eval
from machine_fault_triage.ports import TextSplitter
from machine_fault_triage.prompts import CURRENT_PROMPT_VERSION
from machine_fault_triage.selection import HybridPageSelector, KeywordPageSelector

if TYPE_CHECKING:
    from machine_fault_triage.ports import Embedder, StructuredModel

_USAGE_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return its exit code."""
    arguments = _parser().parse_args(argv)
    try:
        output = _COMMANDS[arguments.command](arguments)
    except TriageError as error:
        sys.stderr.write(f"error: {error}\n")
        return _USAGE_ERROR
    sys.stdout.write(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="machine-fault-triage")
    groups = parser.add_subparsers(required=True)
    library = groups.add_parser("library", help="manage a folder of PDF manuals")
    sync = library.add_subparsers(dest="command", required=True).add_parser(
        "sync", help="read new and changed manuals into the registry and drop removed ones"
    )
    sync.add_argument(
        "--folder", type=Path, required=True, help="manuals as TENANT/DEVICE/NAME.pdf"
    )
    sync.add_argument("--registry", type=Path, required=True, help="registry file to update")
    sync.add_argument(
        "--ocr",
        metavar="MODULE:CALLABLE",
        help="import path of a callable that returns the engine recognising pages without text",
    )
    sync.add_argument(
        "--ocr-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument passed to the engine callable, repeatable",
    )
    sync.add_argument(
        "--embedding-cache",
        type=Path,
        metavar="FILE",
        help="also delete the stored vectors of pages that no manual holds any more",
    )
    evaluation = groups.add_parser("eval", help="measure a model on a dataset")
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
        "--evidence-limit",
        type=int,
        default=1_000_000,
        help="most characters of evidence sent in one model call",
    )
    run.add_argument(
        "--report-limit",
        type=int,
        default=20_000,
        help="most characters of one fault report",
    )
    run.add_argument(
        "--retries",
        type=int,
        default=1,
        help="how often an invalid or truncated answer is asked for again",
    )
    run.add_argument(
        "--price",
        type=float,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="model price in US dollars per million input and output tokens",
    )
    run.add_argument(
        "--cached-price",
        type=float,
        metavar="CACHED_INPUT",
        help="with --price: price per million input tokens read from the prompt cache",
    )
    run.add_argument(
        "--embedding-price",
        type=float,
        metavar="EMBEDDING",
        help="with --price: price per million embedding tokens",
    )
    run.add_argument(
        "--pages",
        type=int,
        metavar="N",
        help="send only the N pages that best match each report instead of every page",
    )
    run.add_argument(
        "--neighbours",
        type=int,
        metavar="N",
        default=0,
        help="with --pages: also send N pages before and after each chosen page",
    )
    run.add_argument(
        "--chunk-size",
        type=int,
        metavar="CHARACTERS",
        help="with --pages: rank pieces of pages of at most this many characters instead of"
        " whole pages; whole pages are still sent",
    )
    run.add_argument(
        "--chunk-overlap",
        type=int,
        metavar="CHARACTERS",
        default=0,
        help="with --chunk-size: cut fixed pieces that overlap by this many characters",
    )
    run.add_argument(
        "--photo-reader",
        metavar="MODULE:CALLABLE",
        help="import path of a callable that returns the reader of the photos in the cases",
    )
    run.add_argument(
        "--photo-reader-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument for the photo reader callable; repeat for several",
    )
    run.add_argument(
        "--embedder",
        help="with --pages: also match pages by meaning, using the embedder returned by"
        " this module:callable",
    )
    run.add_argument(
        "--embedder-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="keyword argument for the embedder callable; repeat for several",
    )
    run.add_argument(
        "--embedding-cache",
        type=Path,
        metavar="FILE",
        help="with --embedder: keep page vectors in this file between runs",
    )
    run.add_argument(
        "--prompt",
        type=int,
        metavar="VERSION",
        default=CURRENT_PROMPT_VERSION,
        help="prompt version to run, to repeat a past evaluation",
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
    pages = _selector(arguments)
    model: StructuredModel = _load("model", arguments.model, arguments.option)
    reader = (
        _load("photo reader", arguments.photo_reader, arguments.photo_reader_option)
        if arguments.photo_reader
        else None
    )
    report = run_eval(
        load_dataset(arguments.dataset),
        model,
        model_label=arguments.label or arguments.model,
        repeats=arguments.repeats,
        retries=arguments.retries,
        evidence_limit=arguments.evidence_limit,
        report_limit=arguments.report_limit,
        pricing=_pricing(arguments),
        pages=pages,
        page_selection=None if pages is None else pages.description,
        prompt_version=arguments.prompt,
        photo_reader=reader,
        photo_reader_label=" ".join((arguments.photo_reader or "", *arguments.photo_reader_option)),
    )
    arguments.out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return render_summary(report)


def _sync(arguments: argparse.Namespace) -> str:
    try:
        folder = importlib.import_module("machine_fault_triage.folder")
    except ImportError as error:
        message = f"library sync needs the pdf extra: {error}"
        raise InvalidRequestError(message) from error
    ocr = (
        _load("recognition engine", arguments.ocr, arguments.ocr_option) if arguments.ocr else None
    )
    report = folder.FolderLibrary(arguments.folder, arguments.registry, ocr=ocr).sync()
    lines = [
        f"{state:<9} {name}"
        for state in ("added", "changed", "removed", "failed", "unchanged")
        for name in getattr(report, state)
    ]
    lines.extend(f"note      {note}" for note in report.notes)
    if arguments.embedding_cache and report.forgotten_pages:
        deleted = forget_pages(arguments.embedding_cache, report.forgotten_pages)
        lines.append(f"{deleted} stored vectors deleted")
    return "".join(f"{line}\n" for line in lines) or "the folder holds no manuals\n"


def _pricing(arguments: argparse.Namespace) -> Pricing | None:
    if arguments.price is None:
        if arguments.cached_price is not None or arguments.embedding_price is not None:
            message = "--cached-price and --embedding-price need --price"
            raise InvalidRequestError(message)
        return None
    try:
        return Pricing(
            input=arguments.price[0],
            output=arguments.price[1],
            cached_input=arguments.cached_price,
            embedding=arguments.embedding_price,
        )
    except ValidationError as error:
        message = f"a price cannot be negative: {error}"
        raise InvalidRequestError(message) from error


def _compare(arguments: argparse.Namespace) -> str:
    return render_comparison(
        compare_runs(_read_report(arguments.baseline), _read_report(arguments.candidate))
    )


def _selector(arguments: argparse.Namespace) -> KeywordPageSelector | HybridPageSelector | None:
    refinements = (
        arguments.neighbours,
        arguments.embedder,
        arguments.embedding_cache,
        arguments.chunk_size,
    )
    if arguments.pages is None:
        if any(refinements):
            message = "--neighbours, --chunk-size, --embedder and --embedding-cache need --pages"
            raise InvalidRequestError(message)
        return None
    splitter = _splitter(arguments)
    if arguments.embedder is None:
        if arguments.embedding_cache:
            message = "--embedding-cache needs --embedder"
            raise InvalidRequestError(message)
        return KeywordPageSelector(
            arguments.pages, neighbours=arguments.neighbours, splitter=splitter
        )
    embedder: Embedder = _load("embedder", arguments.embedder, arguments.embedder_option)
    name = " ".join((arguments.embedder, *arguments.embedder_option))
    if arguments.embedding_cache:
        embedder = CachedEmbedder(embedder, arguments.embedding_cache, model=name)
    return HybridPageSelector(
        embedder,
        arguments.pages,
        neighbours=arguments.neighbours,
        embedder_name=name,
        splitter=splitter,
    )


def _splitter(arguments: argparse.Namespace) -> TextSplitter | None:
    if arguments.chunk_size is None:
        if arguments.chunk_overlap:
            message = "--chunk-overlap needs --chunk-size"
            raise InvalidRequestError(message)
        return None
    try:
        chunking = importlib.import_module("machine_fault_triage.adapters.chonkie")
    except ImportError as error:
        message = f"--chunk-size needs the chunking extra: {error}"
        raise InvalidRequestError(message) from error
    splitter: TextSplitter = chunking.ChonkieSplitter(
        arguments.chunk_size, overlap=arguments.chunk_overlap
    )
    return splitter


def _load(kind: str, path: str, options: Sequence[str]) -> Any:  # noqa: ANN401
    module_name, _, attribute = path.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        return factory(**_keyword_arguments(options))
    except (ImportError, AttributeError, ValueError, TypeError) as error:
        message = f"cannot load the {kind} from {path!r}: {error}"
        raise InvalidRequestError(message) from error


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


_COMMANDS = {"run": _run, "compare": _compare, "sync": _sync}
