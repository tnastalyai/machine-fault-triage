"""Evaluation cases and the dataset directory that holds them."""

import hashlib
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from machine_fault_triage.errors import DatasetError
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.inputs import Document, FaultReport, ServiceRecord
from machine_fault_triage.models import TriageStatus


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


class Expectation(_Spec):
    """What a correct result looks like for one case.

    ``cause_terms`` is a list of groups. The top cause is correct when every
    group has at least one term that occurs in its text. ``forbidden`` lists
    strings that only a successful prompt injection would put into the result.
    """

    status: TriageStatus
    cause_terms: tuple[tuple[str, ...], ...] = ()
    evidence: tuple[ExpectedEvidence, ...] = ()
    safety_warning: bool = False
    forbidden: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _expectation_is_decidable(self) -> Self:
        if self.status not in {TriageStatus.OK, TriageStatus.NOT_ENOUGH_DATA}:
            message = "a case expects either ok or not_enough_data"
            raise ValueError(message)
        answerable = self.status is TriageStatus.OK
        if answerable != bool(self.cause_terms) or not all(self.cause_terms):
            message = "exactly the ok cases name the terms of the expected cause"
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
    documents: tuple[str, ...] = ()
    records: tuple[ServiceRecord, ...] = ()
    expect: Expectation


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
        for expected in case.expect.evidence:
            if not _is_supplied(expected, case, documents):
                message = f"case {case.id} expects evidence it does not supply: {expected!r}"
                raise ValueError(message)


def load_dataset(root: Path) -> Dataset:
    """Load a dataset from ``root/documents/*.toml`` and ``root/cases/*.toml``.

    Raises:
        DatasetError: If a file cannot be read or parsed, or the cases contradict
            the documents and records they refer to.
    """
    document_paths = sorted((root / "documents").glob("*.toml"))
    case_paths = sorted((root / "cases").glob("*.toml"))
    documents = [_load(path, Document) for path in document_paths]
    cases = [_load(path, EvalCase) for path in case_paths]
    try:
        return Dataset(
            name=root.resolve().name,
            digest=_digest(root, [*document_paths, *case_paths]),
            documents=tuple(documents),
            cases=tuple(cases),
        )
    except ValidationError as error:
        message = f"{root}: {error}"
        raise DatasetError(message) from error


def _load[SpecT: BaseModel](path: Path, spec: type[SpecT]) -> SpecT:
    try:
        return spec.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        message = f"{path}: {error}"
        raise DatasetError(message) from error


def _digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
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
