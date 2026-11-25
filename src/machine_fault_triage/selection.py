"""Choosing the document pages that are sent to the model."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

from machine_fault_triage.errors import InvalidRequestError
from machine_fault_triage.inputs import Document, FaultReport
from machine_fault_triage.ports import Embedder, Selection, TextSplitter

_TOKEN = re.compile(r"\w+(?:[.\-/]\w+)*")
_CODE = re.compile(r"[^\W\d_]{1,2}\d{1,4}")
_STEM_LENGTH = 5
_TERM_SATURATION = 1.5
_LENGTH_WEIGHT = 0.75
_RANK_DAMPING = 60

type _Key = tuple[str, int]


class _Piece(NamedTuple):
    page: _Key
    text: str


class KeywordPageSelector:
    """Keeps the pages that share the most telling words with the fault report.

    Pages are ranked with BM25 over all documents of the device. Words are cut
    to their first five characters, a crude stemmer that needs no dictionary;
    fault codes such as ``A.11`` are also matched whole, and as ``A11``, the
    way a customer often types them. Matching is lexical: a report in another
    language than the documents, or one that describes the fault in other
    words than the manual, finds little or nothing.
    """

    def __init__(
        self, limit: int = 20, *, neighbours: int = 0, splitter: TextSplitter | None = None
    ) -> None:
        """Choose how many pages to keep at most, across all documents.

        With ``neighbours``, each chosen page brings that many pages before
        and after it from the same document, counted within the limit. That
        helps when a table runs on to the next page, and costs places: with a
        small limit the neighbours push out pages that matched. With
        ``splitter``, the pieces it cuts are ranked instead of whole pages,
        and a page is as good as its best piece; whole pages are still kept.

        Raises:
            InvalidRequestError: If ``limit`` is less than one or
                ``neighbours`` is negative.
        """
        _check(limit, neighbours)
        self._limit = limit
        self._neighbours = neighbours
        self._splitter = splitter

    @property
    def description(self) -> str:
        """A short name of this selection for evaluation reports."""
        return _describe("keyword", self._limit, self._neighbours, self._splitter)

    def rank(self, report: FaultReport, documents: Sequence[Document]) -> list[_Key]:
        """Return the kept pages as ``(document id, page number)``, best first."""
        pieces = _pieces(documents, self._splitter)
        ranked = _pages(pieces, _keyword_order(report, pieces))
        return _within(documents, ranked, self._limit, self._neighbours)

    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        """Return the documents with only their best-matching pages, in page order.

        Documents that fit within the limit as they are come back unchanged.
        """
        if _page_count(documents) <= self._limit:
            return Selection(documents)
        return Selection(_only(documents, self.rank(report, documents)))


class HybridPageSelector:
    """Keeps the pages that match the fault report by wording or by meaning.

    Two rankings are made: BM25, as in ``KeywordPageSelector``, and cosine
    similarity between the vectors of the report and of each page. The top of
    each list, twice the limit, is merged with reciprocal rank fusion: a page
    scores by its position on each list, so a page near the top of either list
    is kept and a page on both rises. The keyword ranking finds exact fault
    codes and part numbers; the vectors find a page that describes the fault
    in other words or in another language.
    """

    def __init__(
        self,
        embedder: Embedder,
        limit: int = 20,
        *,
        neighbours: int = 0,
        embedder_name: str = "unnamed embedder",
        splitter: TextSplitter | None = None,
    ) -> None:
        """Choose the embedder and how many pages to keep at most.

        ``embedder_name`` appears in ``description``, so that evaluation runs
        with different embedding models can be told apart. With ``splitter``,
        the pieces it cuts are ranked and embedded instead of whole pages;
        whole pages are still kept.

        Raises:
            InvalidRequestError: If ``limit`` is less than one or
                ``neighbours`` is negative.
        """
        _check(limit, neighbours)
        self._embedder = embedder
        self._embedder_name = embedder_name
        self._limit = limit
        self._neighbours = neighbours
        self._splitter = splitter

    @property
    def description(self) -> str:
        """A short name of this selection for evaluation reports."""
        kind = f"hybrid with {self._embedder_name}"
        return _describe(kind, self._limit, self._neighbours, self._splitter)

    def rank(self, report: FaultReport, documents: Sequence[Document]) -> list[_Key]:
        """Return the kept pages as ``(document id, page number)``, best first.

        Calls the embedder even when every page fits within the limit.
        """
        return self._ranked(report, documents)[0]

    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        """Return the documents with only their best-matching pages, in page order.

        Documents that fit within the limit as they are come back unchanged,
        without a call to the embedder. Otherwise the selection reports the
        tokens the embedder billed for the report and for pages it had not
        embedded before.
        """
        if _page_count(documents) <= self._limit:
            return Selection(documents)
        ranked, tokens = self._ranked(report, documents)
        return Selection(_only(documents, ranked), tokens)

    def _ranked(self, report: FaultReport, documents: Sequence[Document]) -> tuple[list[_Key], int]:
        pieces = _pieces(documents, self._splitter)
        depth = 2 * self._limit
        semantic, tokens = self._semantic_order(report, pieces)
        fused: dict[int, float] = {}
        for order in (_keyword_order(report, pieces), semantic):
            for position, index in enumerate(order[:depth], start=1):
                fused[index] = fused.get(index, 0.0) + 1 / (_RANK_DAMPING + position)
        ranked = _pages(pieces, sorted(fused, key=lambda index: -fused[index]))
        return _within(documents, ranked, self._limit, self._neighbours), tokens

    def _semantic_order(
        self, report: FaultReport, pieces: Sequence[_Piece]
    ) -> tuple[list[int], int]:
        written = [index for index, piece in enumerate(pieces) if piece.text.strip()]
        pages = self._embedder.embed_pages([pieces[index].text for index in written])
        asked = self._embedder.embed_query(report.description)
        (query,) = asked.vectors
        similarity = {
            index: _cosine(query, vector)
            for index, vector in zip(written, pages.vectors, strict=True)
        }
        order = sorted(similarity, key=lambda index: -similarity[index])
        return order, pages.tokens + asked.tokens


def _check(limit: int, neighbours: int) -> None:
    if limit < 1:
        message = "limit must be at least 1"
        raise InvalidRequestError(message)
    if neighbours < 0:
        message = "neighbours must not be negative"
        raise InvalidRequestError(message)


def _describe(kind: str, limit: int, neighbours: int, splitter: TextSplitter | None) -> str:
    around = f", {neighbours} neighbouring on each side" if neighbours else ""
    ranked = f", ranked by {splitter.description}" if splitter else ""
    return f"{kind}, at most {limit} pages{around}{ranked}"


def _page_count(documents: Sequence[Document]) -> int:
    return sum(len(document.pages) for document in documents)


def _pieces(documents: Sequence[Document], splitter: TextSplitter | None) -> list[_Piece]:
    pieces: list[_Piece] = []
    for document in documents:
        for page in document.pages:
            texts = splitter.split(page.text) if splitter else ()
            pieces.extend(_Piece((document.id, page.number), text) for text in texts or [page.text])
    return pieces


def _pages(pieces: Sequence[_Piece], order: Sequence[int]) -> list[_Key]:
    return list(dict.fromkeys(pieces[index].page for index in order))


def _keyword_order(report: FaultReport, pieces: Sequence[_Piece]) -> list[int]:
    terms = [_terms(piece.text) for piece in pieces]
    query = set(_terms(report.description))
    containing = Counter(term for found in terms for term in query.intersection(found))
    mean_length = sum(len(found) for found in terms) / len(terms)
    scores = [_score(Counter(found), query, containing, len(terms), mean_length) for found in terms]
    matched = [index for index, value in enumerate(scores) if value > 0]
    return sorted(matched, key=lambda index: -scores[index])


def _within(
    documents: Sequence[Document], ranked: Sequence[_Key], limit: int, neighbours: int
) -> list[_Key]:
    existing = {(document.id, page.number) for document in documents for page in document.pages}
    kept: dict[_Key, None] = {}
    for document_id, number in ranked:
        around = sorted(
            range(number - neighbours, number + neighbours + 1), key=lambda n: abs(n - number)
        )
        for candidate in around:
            if len(kept) < limit and (document_id, candidate) in existing:
                kept[(document_id, candidate)] = None
    return list(kept)


def _only(documents: Sequence[Document], kept: Sequence[_Key]) -> list[Document]:
    pages = set(kept)
    return [
        document.model_copy(
            update={"pages": tuple(p for p in document.pages if (document.id, p.number) in pages)}
        )
        for document in documents
        if any(document_id == document.id for document_id, _ in pages)
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norm = math.hypot(*left) * math.hypot(*right)
    return sum(a * b for a, b in zip(left, right, strict=True)) / norm if norm else 0.0


def _terms(text: str) -> list[str]:
    terms = []
    for token in _TOKEN.findall(text.casefold()):
        parts = re.split(r"[.\-/]", token)
        if len(parts) > 1:
            terms.append(token)
            if _CODE.fullmatch("".join(parts)):
                terms.append("".join(parts))
        terms.extend(part[:_STEM_LENGTH] for part in parts)
    return terms


def _score(
    counts: Counter[str], query: set[str], containing: Counter[str], total: int, mean_length: float
) -> float:
    length = sum(counts.values())
    score = 0.0
    for term in query:
        frequency = counts[term]
        if not frequency:
            continue
        rarity = math.log(1 + (total - containing[term] + 0.5) / (containing[term] + 0.5))
        norm = _TERM_SATURATION * (1 - _LENGTH_WEIGHT + _LENGTH_WEIGHT * length / mean_length)
        score += rarity * frequency * (_TERM_SATURATION + 1) / (frequency + norm)
    return score
