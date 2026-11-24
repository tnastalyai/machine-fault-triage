"""Turns model claims into causes whose citations were checked against the evidence."""

from collections.abc import Sequence

from machine_fault_triage.drafts import DraftCause, DraftCitation, DraftDocumentCitation
from machine_fault_triage.inputs import Document, ServiceRecord
from machine_fault_triage.models import (
    Cause,
    Citation,
    Confidence,
    DocumentCitation,
    EvidenceLevel,
    HistoryCitation,
)


def ground(
    drafts: Sequence[DraftCause],
    documents: Sequence[Document],
    records: Sequence[ServiceRecord],
) -> tuple[Cause, ...]:
    """Build causes from model drafts, keeping only what the evidence supports.

    A document citation is verified when its quote occurs on the cited page. A
    history citation is verified when the record was part of the supplied
    evidence. Citations that point at unknown documents or records are dropped,
    and a cause left without a verified citation is reduced to low confidence.
    """
    documents_by_id = {document.id: document for document in documents}
    records_by_id = {record.id: record for record in records}
    return tuple(_ground_cause(draft, documents_by_id, records_by_id) for draft in drafts)


def evidence_level(causes: Sequence[Cause]) -> EvidenceLevel:
    """Name the strongest kind of verified evidence behind the causes."""
    verified = [citation for cause in causes for citation in cause.citations if citation.verified]
    if any(isinstance(citation, DocumentCitation) for citation in verified):
        return EvidenceLevel.MANUAL
    if verified:
        return EvidenceLevel.HISTORY
    return EvidenceLevel.GENERAL_KNOWLEDGE if causes else EvidenceLevel.NONE


def contains_quote(text: str, quote: str) -> bool:
    """Tell whether the quote occurs in the text, ignoring case and whitespace runs."""
    return _normalize(quote) in _normalize(text)


def _ground_cause(
    draft: DraftCause,
    documents: dict[str, Document],
    records: dict[str, ServiceRecord],
) -> Cause:
    citations = tuple(
        citation
        for citation in (_ground_citation(item, documents, records) for item in draft.citations)
        if citation is not None
    )
    grounded = any(citation.verified for citation in citations)
    return Cause(
        title=draft.title,
        confidence=draft.confidence if grounded else Confidence.LOW,
        why=draft.why,
        checks=draft.checks,
        fix=draft.fix,
        citations=citations,
    )


def _ground_citation(
    draft: DraftCitation,
    documents: dict[str, Document],
    records: dict[str, ServiceRecord],
) -> Citation | None:
    if isinstance(draft, DraftDocumentCitation):
        return _ground_document_citation(draft, documents.get(draft.document_id))
    record = records.get(draft.record_id)
    if record is None:
        return None
    return HistoryCitation(
        record_id=record.id, recorded_on=record.recorded_on, scope=record.scope, verified=True
    )


def _ground_document_citation(
    draft: DraftDocumentCitation, document: Document | None
) -> DocumentCitation | None:
    if document is None or draft.page < 1 or not draft.quote.strip():
        return None
    page = document.page(draft.page)
    return DocumentCitation(
        document_id=document.id,
        document_title=document.title,
        page=draft.page,
        quote=draft.quote,
        verified=page is not None and contains_quote(page.text, draft.quote),
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
