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
from machine_fault_triage.evals.report import (
    CaseOutcome,
    MetricSummary,
    RepeatOutcome,
    RunReport,
    summarize,
)
from machine_fault_triage.evals.runner import run_eval
from machine_fault_triage.evals.scoring import Metric, score

__all__ = [
    "CaseOutcome",
    "Dataset",
    "EvalCase",
    "Expectation",
    "ExpectedDocumentEvidence",
    "ExpectedEvidence",
    "ExpectedHistoryEvidence",
    "Metric",
    "MetricSummary",
    "RepeatOutcome",
    "RunReport",
    "load_dataset",
    "run_eval",
    "score",
    "summarize",
]
