"""Runs every case of a dataset through the triage pipeline and scores the results."""

import time
from collections.abc import Sequence
from importlib.metadata import version
from typing import NamedTuple

from pydantic import BaseModel

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import InvalidRequestError
from machine_fault_triage.evals.cases import Dataset, EvalCase
from machine_fault_triage.evals.report import CaseOutcome, Pricing, RepeatOutcome, RunReport
from machine_fault_triage.evals.scoring import score
from machine_fault_triage.inputs import (
    Document,
    FaultReport,
    ServiceRecord,
    without_superseded,
)
from machine_fault_triage.pipeline import (
    check_evidence_size,
    check_report_size,
    evidence_request,
    run_triage,
)
from machine_fault_triage.ports import (
    Generation,
    GenerationRequest,
    PageRanker,
    PageSelector,
    Selection,
    StructuredModel,
)
from machine_fault_triage.prompts import CURRENT_PROMPT_VERSION, instructions, prompt_digest

_TENANT_ID = "evaluation"


def run_eval(  # noqa: PLR0913
    dataset: Dataset,
    model: StructuredModel,
    *,
    model_label: str,
    repeats: int = 1,
    pricing: Pricing | None = None,
    pages: PageSelector | None = None,
    page_selection: str | None = None,
    prompt_version: int = CURRENT_PROMPT_VERSION,
    evidence_limit: int = 1_000_000,
    report_limit: int = 20_000,
    retries: int = 1,
) -> RunReport:
    """Run each case ``repeats`` times against the model and score every result.

    Each run is timed, and its token usage is kept when the model reports it.
    ``pricing`` is stored in the report so that the tokens can be priced.
    Embedding tokens are those page selection spent on the pages it sent.
    Ranking the pages once more for the retrieval metrics is not counted.
    With ``pages`` the model sees only the pages that selector chooses for
    each case; ``page_selection`` names the selection in the report. Without
    ``pages`` the model sees every page of the case's documents.

    ``prompt_version`` runs an older version of the instructions.

    ``evidence_limit`` caps the characters of evidence sent in one call.
    Without ``pages`` every case is checked against it before the first
    model call, so that a run does not stop halfway through a paid dataset.
    ``report_limit`` caps the characters of a fault report and is checked for
    every case up front as well.

    ``retries`` is how often an invalid or truncated answer is asked for again.

    Raises:
        InvalidRequestError: If ``repeats`` is less than one, if ``retries``
            is negative, if no prompt has the given version, or if a case
            exceeds ``report_limit`` or ``evidence_limit``.
    """
    if repeats < 1 or retries < 0:
        message = "repeats must be at least 1 and retries must not be negative"
        raise InvalidRequestError(message)
    for case in dataset.cases:
        check_report_size(case.report, report_limit)
    if pages is None:
        for case in dataset.cases:
            documents = without_superseded(dataset.documents_for(case))
            check_evidence_size(
                evidence_request(
                    instructions(prompt_version), case.report, documents, case.records
                ),
                evidence_limit,
            )
    return RunReport(
        dataset=dataset.name,
        dataset_digest=dataset.digest,
        model=model_label,
        library_version=version("machine-fault-triage"),
        prompt_version=prompt_version,
        prompt_digest=prompt_digest(prompt_version),
        page_selection=page_selection,
        repeats=repeats,
        pricing=pricing,
        cases=tuple(
            CaseOutcome(
                case_id=case.id,
                category=case.category,
                repeats=tuple(
                    _run_case(
                        case,
                        dataset,
                        model,
                        _Settings(pages, prompt_version, evidence_limit, report_limit, retries),
                    )
                    for _ in range(repeats)
                ),
            )
            for case in dataset.cases
        ),
    )


class _Settings(NamedTuple):
    pages: PageSelector | None
    prompt_version: int
    evidence_limit: int
    report_limit: int
    retries: int


def _run_case(
    case: EvalCase, dataset: Dataset, model: StructuredModel, settings: _Settings
) -> RepeatOutcome:
    pages = settings.pages
    recorder = _Recorder(model)
    started = time.perf_counter()
    documents = without_superseded(dataset.documents_for(case))
    selection = Selection(documents) if pages is None else pages.select(case.report, documents)
    ranked = pages.rank(case.report, documents) if isinstance(pages, PageRanker) else None
    evidence = _CaseEvidence(documents, case.records)
    result = run_triage(
        case.report,
        tenant_id=_TENANT_ID,
        device_id=case.id,
        model=recorder,
        documents=evidence,
        history=evidence,
        pages=_Selected(selection),
        prompt_version=settings.prompt_version,
        evidence_limit=settings.evidence_limit,
        report_limit=settings.report_limit,
        retries=settings.retries,
    )
    return RepeatOutcome(
        result=result,
        scores=score(case, recorder.draft, result, selection.documents, ranked),
        seconds=time.perf_counter() - started,
        usage=result.usage,
    )


class _Selected:
    def __init__(self, selection: Selection) -> None:
        self._selection = selection

    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        del report, documents
        return self._selection


class _CaseEvidence:
    def __init__(self, documents: Sequence[Document], records: Sequence[ServiceRecord]) -> None:
        self._documents = documents
        self._records = records

    def documents_for(self, *, tenant_id: str, device_id: str) -> Sequence[Document]:
        del tenant_id, device_id
        return self._documents

    def records_for(self, *, tenant_id: str, device_id: str) -> Sequence[ServiceRecord]:
        del tenant_id, device_id
        return self._records


class _Recorder:
    def __init__(self, model: StructuredModel) -> None:
        self._model = model
        self.draft: DraftTriage | None = None

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        generation = self._model.generate(request, output_type)
        if isinstance(generation.output, DraftTriage):
            self.draft = generation.output
        return generation
