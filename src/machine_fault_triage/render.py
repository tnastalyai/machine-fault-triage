"""A triage result laid out as plain text for a technician."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from machine_fault_triage.models import (
    Cause,
    Citation,
    Confidence,
    DocumentCitation,
    HistoryScope,
    PageSource,
    TriageResult,
    TriageStatus,
)


@dataclass(frozen=True, kw_only=True)
class Labels:
    """The fixed words of a rendered result. The defaults are English.

    The result itself is written in the language of the fault report; pass
    labels in that language to get a page in one language.
    """

    not_enough_data: str = "Not enough evidence for a diagnosis"
    refused: str = "The model declined to answer"
    failed: str = "The run failed"
    safety: str = "Safety"
    reported: str = "Reported by the customer"
    unreported_figures: str = "Not found in the report or its photos"
    photos: str = "Read from the photos, not checked"
    summary: str = "Summary"
    causes: str = "Probable causes"
    high: str = "high confidence"
    medium: str = "medium confidence"
    low: str = "low confidence"
    why: str = "Why"
    checks: str = "Checks"
    fix: str = "Fix"
    sources: str = "Sources"
    page: str = "p."
    not_verified: str = "NOT VERIFIED: the quote was not found in the document"
    recognised: str = "recognised from a scan, may hold misread characters"
    cited_page: str = "the model named p."
    record: str = "service record"
    same_model: str = "another device of the same model"
    parts: str = "Parts to bring"
    questions: str = "Questions for the customer"
    missing: str = "Missing information"
    unsupported_figures: str = "Numbers in this answer that are not in the evidence"


def render_text(result: TriageResult, *, labels: Labels | None = None) -> str:
    """Lay a result out as plain text: safety first, then the report, causes and the rest.

    Nothing that weakens the answer is hidden: a quote that was not verified,
    a page recognised from a scan, a page number the library corrected, low
    confidence, numbers that are not in the evidence and the reason of a
    failed run are all printed. Empty sections are left out.

    The text of the result comes from a model and from documents. Line breaks
    inside it are replaced by spaces, so that it cannot imitate a heading of
    this layout. The output is plain text, not Markdown or HTML: escape it
    before putting it into either.
    """
    words = labels or Labels()
    confidence = {
        Confidence.HIGH: words.high,
        Confidence.MEDIUM: words.medium,
        Confidence.LOW: words.low,
    }
    status = {
        TriageStatus.NOT_ENOUGH_DATA: words.not_enough_data,
        TriageStatus.REFUSED: words.refused,
        TriageStatus.FAILED: words.failed,
    }
    reported = [f"- {_line(fact)}" for fact in result.report_facts]
    if result.unreported_figures:
        reported.append(f"! {words.unreported_figures}: {_joined(result.unreported_figures)}")
    causes = [
        line
        for number, cause in enumerate(result.causes, start=1)
        for line in _cause(number, cause, confidence[cause.confidence], words)
    ]
    headline = status.get(result.status)
    sections = [
        _section(words.safety, (f"! {_line(warning)}" for warning in result.safety_warnings)),
        [headline.upper(), *([_line(result.failure)] if result.failure else [])]
        if headline
        else [],
        _section(words.reported, reported),
        _section(words.photos, (f"- {_line(reading)}" for reading in result.photo_readings)),
        _section(words.summary, [_line(result.summary)]),
        _section(words.causes, causes),
        _section(words.parts, (f"- {_line(part)}" for part in result.parts_to_bring)),
        _section(words.questions, (f"- {_line(item)}" for item in result.questions_for_customer)),
        _section(words.missing, (f"- {_line(item)}" for item in result.missing)),
        _section(words.unsupported_figures, [_joined(result.unsupported_figures)]),
    ]
    return "\n\n".join("\n".join(section) for section in sections if section) + "\n"


def _section(title: str, lines: Iterable[str]) -> list[str]:
    body = [line for line in lines if line.strip("-! ")]
    return [title.upper(), *body] if body else []


def _cause(number: int, cause: Cause, confidence: str, words: Labels) -> list[str]:
    lines = [
        f"{number}. {_line(cause.title)} [{confidence}]",
        f"   {words.why}: {_line(cause.why)}",
    ]
    if cause.checks:
        lines.append(f"   {words.checks}:")
        lines.extend(f"     {step}) {_line(check)}" for step, check in enumerate(cause.checks, 1))
    if cause.fix:
        lines.append(f"   {words.fix}: {_line(cause.fix)}")
    if cause.citations:
        lines.append(f"   {words.sources}:")
        lines.extend(f"     - {_citation(citation, words)}" for citation in cause.citations)
    return lines


def _citation(citation: Citation, words: Labels) -> str:
    if not isinstance(citation, DocumentCitation):
        other = f" ({words.same_model})" if citation.scope is HistoryScope.MODEL else ""
        return f"{words.record} {_line(citation.record_id)}, {citation.recorded_on}{other}"
    flags = [
        "" if citation.verified else words.not_verified,
        words.recognised if citation.source is PageSource.OCR else "",
        f"{words.cited_page} {citation.cited_page}" if citation.cited_page else "",
    ]
    noted = "".join(f" [{flag}]" for flag in flags if flag)
    title = _line(citation.document_title)
    return f'{title}, {words.page} {citation.page}: "{_line(citation.quote)}"{noted}'


def _joined(items: Sequence[str]) -> str:
    return ", ".join(_line(item) for item in items)


def _line(text: str) -> str:
    return " ".join(text.split())
