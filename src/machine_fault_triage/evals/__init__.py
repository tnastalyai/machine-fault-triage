"""Evaluation harness: cases, code-computed metrics, runs and comparisons."""

from machine_fault_triage.evals.cases import (
    Dataset,
    EvalCase,
    Expectation,
    ExpectedDocumentEvidence,
    ExpectedEvidence,
    ExpectedHistoryEvidence,
    load_dataset,
)
from machine_fault_triage.evals.scoring import Metric, score

__all__ = [
    "Dataset",
    "EvalCase",
    "Expectation",
    "ExpectedDocumentEvidence",
    "ExpectedEvidence",
    "ExpectedHistoryEvidence",
    "Metric",
    "load_dataset",
    "score",
]
