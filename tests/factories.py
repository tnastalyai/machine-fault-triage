from collections.abc import Sequence
from datetime import date

from machine_fault_triage import Document, FaultReport, HistoryScope, Page, ServiceRecord
from machine_fault_triage.drafts import (
    DraftCause,
    DraftCitation,
    DraftDocumentCitation,
    DraftHistoryCitation,
    DraftTriage,
)
from machine_fault_triage.models import Confidence

MANUAL_ID = "manual-1"
RECORD_ID = "order-9"
QUOTE = "E7: flow sensor open or short circuit"


def manual() -> Document:
    return Document(
        id=MANUAL_ID,
        title="AquaHeat 12 service manual",
        pages=(
            Page(number=76, text="7.2 Start-up sequence"),
            Page(number=77, text=f"7.3 Error codes\nE6: low pressure\n{QUOTE}\n"),
        ),
    )


def service_record() -> ServiceRecord:
    return ServiceRecord(
        id=RECORD_ID,
        recorded_on=date(2026, 3, 11),
        scope=HistoryScope.DEVICE,
        summary="E7 on display, flow sensor replaced",
        parts_replaced=("Flow sensor",),
    )


def fault_report() -> FaultReport:
    return FaultReport(description="Display shows E7 and the unit stops heating.", language="en")


def manual_citation(quote: str = QUOTE, page: int = 77) -> DraftDocumentCitation:
    return DraftDocumentCitation(document_id=MANUAL_ID, page=page, quote=quote)


def record_citation(record_id: str = RECORD_ID) -> DraftHistoryCitation:
    return DraftHistoryCitation(record_id=record_id)


def draft_cause(
    citations: Sequence[DraftCitation] = (), confidence: Confidence = Confidence.HIGH
) -> DraftCause:
    return DraftCause(
        title="Flow sensor fault",
        confidence=confidence,
        why="E7 maps to the flow sensor.",
        checks=("Measure sensor resistance.",),
        fix="Replace the flow sensor.",
        citations=tuple(citations),
    )


def draft_triage(causes: Sequence[DraftCause]) -> DraftTriage:
    return DraftTriage(
        summary="E7 points to the flow sensor.",
        enough_evidence=True,
        causes=tuple(causes),
        parts_to_bring=("Flow sensor",),
    )


class StaticEvidence:
    def __init__(
        self, documents: Sequence[Document] = (), records: Sequence[ServiceRecord] = ()
    ) -> None:
        self._documents = tuple(documents)
        self._records = tuple(records)
        self.calls: list[tuple[str, str]] = []

    def documents_for(self, *, tenant_id: str, device_id: str) -> Sequence[Document]:
        self.calls.append((tenant_id, device_id))
        return self._documents

    def records_for(self, *, tenant_id: str, device_id: str) -> Sequence[ServiceRecord]:
        self.calls.append((tenant_id, device_id))
        return self._records
