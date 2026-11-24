"""Exceptions raised by the library."""


class TriageError(Exception):
    """Base class for every error raised by this library."""


class InvalidRequestError(TriageError):
    """A triage run was started with arguments that cannot be used."""
