"""Turns model claims into causes whose citations were checked against the evidence."""

import re
from collections.abc import Iterable, Sequence

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

_SHORTEST_QUOTE = 3
_FIGURE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![.,]?\d)(?!\w)")


def ground(
    drafts: Sequence[DraftCause],
    documents: Sequence[Document],
    records: Sequence[ServiceRecord],
) -> tuple[Cause, ...]:
    """Build causes from model drafts, keeping only what the evidence supports.

    A document citation is verified when its quote is at least three words
    long and occurs on the cited page as whole words. A history citation is
    verified when the record was part of the supplied evidence. Citations that
    point at unknown documents or records are dropped, and a cause left without
    a verified citation is reduced to low confidence.
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
    """Tell whether the quote occurs in the text as whole words.

    Case and runs of whitespace are ignored. A quote that begins or ends in
    the middle of a word or a number does not count: ``5 bar`` is not found in
    ``15 bar``.
    """
    wanted = _normalize(quote)
    if not wanted:
        return False
    start = r"(?<!\w)" if wanted[0].isalnum() else ""
    end = r"(?!\w)" if wanted[-1].isalnum() else ""
    return re.search(f"{start}{re.escape(wanted)}{end}", _normalize(text)) is not None


def unsupported_figures(answer: Iterable[str], evidence: Iterable[str]) -> tuple[str, ...]:
    """List the numbers in the answer that occur nowhere in the evidence.

    Only numbers of two or more digits, or with a decimal part, are checked;
    a decimal comma and a decimal point are treated alike. Such a number is a
    reason to distrust the sentence it stands in, not proof that it is wrong:
    the model may have added two readings or converted a unit.
    """
    known = set(_figures(evidence))
    return tuple(dict.fromkeys(figure for figure in _figures(answer) if figure not in known))


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
        verified=page is not None
        and len(draft.quote.split()) >= _SHORTEST_QUOTE
        and contains_quote(page.text, draft.quote),
    )


def _figures(texts: Iterable[str]) -> list[str]:
    return [
        match.replace(",", ".")
        for text in texts
        for match in _FIGURE.findall(text)
        if len(match) > 1
    ]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
