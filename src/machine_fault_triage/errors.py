"""Exceptions raised by the library."""


class TriageError(Exception):
    """Base class for every error raised by this library."""


class InvalidRequestError(TriageError):
    """A triage run was started with arguments that cannot be used."""


class DatasetError(TriageError):
    """An evaluation dataset could not be loaded or contradicts itself."""


class IncomparableRunsError(TriageError):
    """Two evaluation runs cannot be compared because they used different datasets."""
