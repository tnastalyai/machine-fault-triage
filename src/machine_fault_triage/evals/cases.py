"""Evaluation cases and the dataset directory that holds them."""

import hashlib
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from machine_fault_triage.errors import DatasetError
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.inputs import Document, FaultReport, ServiceRecord
from machine_fault_triage.models import TriageStatus

_PHOTO_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExpectedDocumentEvidence(_Spec):
    """A document page the top cause must quote, identified by a phrase in the quote."""

    type: Literal["document"] = "document"
    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    anchor: str = Field(min_length=1)


class ExpectedHistoryEvidence(_Spec):
    """A service record the top cause must cite."""

    type: Literal["history"] = "history"
    record_id: str = Field(min_length=1)


ExpectedEvidence = Annotated[
    ExpectedDocumentEvidence | ExpectedHistoryEvidence, Field(discriminator="type")
]


class ExpectedProblem(_Spec):
    """One of several problems in a report, with the cause and evidence expected for it."""

    cause_terms: tuple[tuple[str, ...], ...] = Field(min_length=1)
    evidence: tuple[ExpectedEvidence, ...] = ()


class Expectation(_Spec):
    """What a correct result looks like for one case.

    ``cause_terms`` is a list of groups. The top cause is correct when every
    group has at least one term that starts a word in its text. ``forbidden``
    lists strings that only a successful prompt injection would put into the
    result; a case that lists any is an attack, and resisting it also means
    ending with the expected status.

    A report that describes several problems lists them under ``problems``
    instead of ``cause_terms`` and ``evidence``: each needs a cause of its own,
    in any position. ``not_covered`` holds terms that name a reported problem
    the evidence says nothing about: no cause may mention them, and the
    summary, the missing information or a question must.

    ``other_models`` lists what the answer would say only if it read the
    values of another model of the device, such as that model's limit with
    its unit. None may appear in the result outside its quotes: a quoted
    table row holds the values of every model.

    ``report_keeps`` lists what the restated report must still say, such as
    the code, and ``report_omits`` what it must have left out, such as a
    phone number or a greeting.

    ``photo_shows`` lists what the reading of the report's photos must hold,
    such as the code on the display. It scores the photo reader on its own,
    whatever the answer then makes of the reading.
    """

    status: TriageStatus
    cause_terms: tuple[tuple[str, ...], ...] = ()
    evidence: tuple[ExpectedEvidence, ...] = ()
    problems: tuple[ExpectedProblem, ...] = ()
    not_covered: tuple[str, ...] = ()
    other_models: tuple[str, ...] = ()
    report_keeps: tuple[str, ...] = ()
    report_omits: tuple[str, ...] = ()
    photo_shows: tuple[str, ...] = ()
    safety_warning: bool = False
    forbidden: tuple[str, ...] = ()

    @property
    def all_evidence(self) -> tuple[ExpectedEvidence, ...]:
        """The evidence expected for the case as a whole and for each of its problems."""
        return (*self.evidence, *(item for problem in self.problems for item in problem.evidence))

    @model_validator(mode="after")
    def _expectation_is_decidable(self) -> Self:
        if self.status not in {TriageStatus.OK, TriageStatus.NOT_ENOUGH_DATA}:
            message = "a case expects either ok or not_enough_data"
            raise ValueError(message)
        answerable = self.status is TriageStatus.OK
        groups = (*self.cause_terms, *(g for problem in self.problems for g in problem.cause_terms))
        if answerable != bool(groups) or not all(groups):
            message = "exactly the ok cases name the terms of the expected cause"
            raise ValueError(message)
        if self.problems and (self.cause_terms or self.evidence):
            message = "a case lists either one expected cause or several problems, not both"
            raise ValueError(message)
        if self.not_covered and not self.problems:
            message = "only a case with several problems can leave one of them not covered"
            raise ValueError(message)
        if not answerable and self.evidence:
            message = "a case that expects abstention cannot expect evidence"
            raise ValueError(message)
        return self


class EvalCase(_Spec):
    """One fault report with the evidence supplied for it and the expected outcome."""

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    report: FaultReport
    device_model: str | None = Field(default=None, min_length=1, max_length=200)
    documents: tuple[str, ...] = ()
    records: tuple[ServiceRecord, ...] = ()
    expect: Expectation

    @model_validator(mode="after")
    def _device_model_is_usable(self) -> Self:
        if self.device_model is not None and not self.device_model.strip():
            message = "a device model must not be blank"
            raise ValueError(message)
        if self.expect.other_models and self.device_model is None:
            message = "a case that lists values of other models must name the device model"
            raise ValueError(message)
        if self.expect.photo_shows and not self.report.photos:
            message = "a case that says what its photos show must have photos"
            raise ValueError(message)
        return self


class Dataset(_Spec):
    """Documents and cases loaded from one directory, checked for consistency."""

    name: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    documents: tuple[Document, ...] = ()
    cases: tuple[EvalCase, ...] = Field(min_length=1)

    def documents_for(self, case: EvalCase) -> tuple[Document, ...]:
        """Return the documents supplied to the model for the given case."""
        return tuple(document for document in self.documents if document.id in case.documents)

    @model_validator(mode="after")
    def _cases_agree_with_the_evidence(self) -> Self:
        _require_unique("document", [document.id for document in self.documents])
        _require_unique("case", [case.id for case in self.cases])
        for case in self.cases:
            self._check_case(case)
        return self

    def _check_case(self, case: EvalCase) -> None:
        documents = {document.id: document for document in self.documents}
        _require_unique(f"record of case {case.id}", [record.id for record in case.records])
        unknown = sorted(set(case.documents) - set(documents))
        if unknown:
            message = f"case {case.id} uses unknown documents: {', '.join(unknown)}"
            raise ValueError(message)
        for expected in case.expect.all_evidence:
            if not _is_supplied(expected, case, documents):
                message = f"case {case.id} expects evidence it does not supply: {expected!r}"
                raise ValueError(message)


def load_dataset(root: Path) -> Dataset:
    """Load a dataset from ``root/documents/*.toml`` and ``root/cases/*.toml``.

    A case names the photos of its report as paths inside the dataset
    directory, under ``photos`` in its ``[report]`` table. They are read
    here, as JPEG, PNG or WebP by their file ending, and count towards the
    digest of the dataset.

    Raises:
        DatasetError: If a file cannot be read or parsed, if a photo lies
            outside the dataset directory or has another file ending, or if
            the cases contradict the documents and records they refer to.
    """
    document_paths = sorted((root / "documents").glob("*.toml"))
    case_paths = sorted((root / "cases").glob("*.toml"))
    documents = [_load(path, Document) for path in document_paths]
    photo_paths: list[Path] = []
    cases = [_load(path, EvalCase, _with_photos(root, photo_paths)) for path in case_paths]
    try:
        return Dataset(
            name=root.resolve().name,
            digest=_digest(root, [*document_paths, *case_paths, *sorted(set(photo_paths))]),
            documents=tuple(documents),
            cases=tuple(cases),
        )
    except ValidationError as error:
        message = f"{root}: {error}"
        raise DatasetError(message) from error


type _Table = dict[str, Any]


def _load[SpecT: BaseModel](
    path: Path, spec: type[SpecT], prepare: Callable[[_Table], _Table] = lambda table: table
) -> SpecT:
    try:
        return spec.model_validate(prepare(tomllib.loads(path.read_text(encoding="utf-8"))))
    except (OSError, tomllib.TOMLDecodeError, ValidationError, ValueError) as error:
        message = f"{path}: {error}"
        raise DatasetError(message) from error


def _with_photos(root: Path, seen: list[Path]) -> Callable[[_Table], _Table]:
    def prepare(table: _Table) -> _Table:
        report = table.get("report")
        if not isinstance(report, dict) or "photos" not in report:
            return table
        photos = []
        for name in report["photos"]:
            path = (root / str(name)).resolve()
            media_type = _PHOTO_TYPES.get(path.suffix.lower())
            if not path.is_relative_to(root.resolve()) or media_type is None:
                message = f"photo {name!r} must be a JPEG, PNG or WebP file inside the dataset"
                raise ValueError(message)
            photos.append({"media_type": media_type, "data": path.read_bytes()})
            seen.append(path)
        return {**table, "report": {**report, "photos": photos}}

    return prepare


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().relative_to(root.resolve()).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_unique(kind: str, ids: list[str]) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        message = f"duplicate {kind} ids: {', '.join(duplicates)}"
        raise ValueError(message)


def _is_supplied(
    expected: ExpectedDocumentEvidence | ExpectedHistoryEvidence,
    case: EvalCase,
    documents: dict[str, Document],
) -> bool:
    if isinstance(expected, ExpectedHistoryEvidence):
        return any(record.id == expected.record_id for record in case.records)
    if expected.document_id not in case.documents:
        return False
    page = documents[expected.document_id].page(expected.page)
    return page is not None and contains_quote(page.text, expected.anchor)
