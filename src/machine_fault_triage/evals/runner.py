"""Runs every case of a dataset through the triage pipeline and scores the results."""

from collections.abc import Sequence
from importlib.metadata import version

from pydantic import BaseModel

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import InvalidRequestError
from machine_fault_triage.evals.cases import Dataset, EvalCase
from machine_fault_triage.evals.report import CaseOutcome, RepeatOutcome, RunReport
from machine_fault_triage.evals.scoring import score
from machine_fault_triage.inputs import Document, ServiceRecord
from machine_fault_triage.pipeline import run_triage
from machine_fault_triage.ports import Generation, GenerationRequest, StructuredModel

_TENANT_ID = "evaluation"


def run_eval(
    dataset: Dataset, model: StructuredModel, *, model_label: str, repeats: int = 1
) -> RunReport:
    """Run each case ``repeats`` times against the model and score every result.

    Raises:
        InvalidRequestError: If ``repeats`` is less than one.
    """
    if repeats < 1:
        message = "repeats must be at least 1"
        raise InvalidRequestError(message)
    return RunReport(
        dataset=dataset.name,
        dataset_digest=dataset.digest,
        model=model_label,
        library_version=version("machine-fault-triage"),
        repeats=repeats,
        cases=tuple(
            CaseOutcome(
                case_id=case.id,
                category=case.category,
                repeats=tuple(_run_case(case, dataset, model) for _ in range(repeats)),
            )
            for case in dataset.cases
        ),
    )


def _run_case(case: EvalCase, dataset: Dataset, model: StructuredModel) -> RepeatOutcome:
    recorder = _Recorder(model)
    evidence = _CaseEvidence(dataset.documents_for(case), case.records)
    result = run_triage(
        case.report,
        tenant_id=_TENANT_ID,
        device_id=case.id,
        model=recorder,
        documents=evidence,
        history=evidence,
    )
    return RepeatOutcome(result=result, scores=score(case, recorder.draft, result))


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
