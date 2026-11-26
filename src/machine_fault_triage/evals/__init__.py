"""Evaluation harness: cases, code-computed metrics, runs and comparisons."""

from machine_fault_triage.evals.cases import (
    Dataset,
    EvalCase,
    Expectation,
    ExpectedDocumentEvidence,
    ExpectedEvidence,
    ExpectedHistoryEvidence,
    ExpectedProblem,
    load_dataset,
)
from machine_fault_triage.evals.compare import Comparison, MetricDelta, compare_runs
from machine_fault_triage.evals.judge import judge_run
from machine_fault_triage.evals.report import (
    CaseOutcome,
    Judgement,
    MetricSummary,
    Pricing,
    RepeatOutcome,
    RunReport,
    Verdict,
    summarize,
)
from machine_fault_triage.evals.runner import run_eval
from machine_fault_triage.evals.scoring import Metric, score

__all__ = [
    "CaseOutcome",
    "Comparison",
    "Dataset",
    "EvalCase",
    "Expectation",
    "ExpectedDocumentEvidence",
    "ExpectedEvidence",
    "ExpectedHistoryEvidence",
    "ExpectedProblem",
    "Judgement",
    "Metric",
    "MetricDelta",
    "MetricSummary",
    "Pricing",
    "RepeatOutcome",
    "RunReport",
    "Verdict",
    "compare_runs",
    "judge_run",
    "load_dataset",
    "run_eval",
    "score",
    "summarize",
]
