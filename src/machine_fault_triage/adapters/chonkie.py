"""Chonkie chunkers behind the ``TextSplitter`` port. Needs the ``chunking`` extra."""

from collections.abc import Sequence

import chonkie

from machine_fault_triage.errors import InvalidRequestError


class ChonkieSplitter:
    """Cuts page text into pieces of at most ``chunk_size`` characters.

    Without ``overlap`` the text is cut recursively: at line breaks first,
    then at sentence ends, and inside a sentence only when it is still too
    long. With ``overlap``, pieces have a fixed size and each one repeats the
    last ``overlap`` characters of the piece before it. Every piece is a
    verbatim part of the page.
    """

    def __init__(self, chunk_size: int = 1300, *, overlap: int = 0) -> None:
        """Choose the piece size and the overlap, both in characters.

        Raises:
            InvalidRequestError: If ``chunk_size`` is less than one, or
                ``overlap`` is negative or not smaller than ``chunk_size``.
        """
        if chunk_size < 1 or not 0 <= overlap < chunk_size:
            message = "chunk_size must be at least 1 and overlap must be smaller than it"
            raise InvalidRequestError(message)
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._chunker: chonkie.TokenChunker | chonkie.RecursiveChunker = (
            chonkie.TokenChunker(chunk_size=chunk_size, chunk_overlap=overlap)
            if overlap
            else chonkie.RecursiveChunker(chunk_size=chunk_size)
        )

    @property
    def description(self) -> str:
        """A short name of the splitting for evaluation reports."""
        if self._overlap:
            return f"fixed pieces of {self._chunk_size} characters, {self._overlap} overlapping"
        return f"recursive pieces of {self._chunk_size} characters"

    def split(self, text: str) -> Sequence[str]:
        """Return the pieces of ``text`` that are not blank, in order."""
        return [chunk.text for chunk in self._chunker.chunk(text) if chunk.text.strip()]
