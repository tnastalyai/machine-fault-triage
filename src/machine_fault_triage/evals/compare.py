"""Paired comparison of two evaluation runs over the same dataset."""

import statistics

from pydantic import BaseModel, ConfigDict

from machine_fault_triage.errors import IncomparableRunsError
from machine_fault_triage.evals.report import RunReport, format_error, standard_error
from machine_fault_triage.evals.scoring import Metric


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetricDelta(_Record):
    """How one metric changed, over the cases where it applied in both runs.

    ``difference`` is candidate minus baseline, averaged over per-case
    differences, and ``standard_error`` describes the spread of those paired
    differences. ``improved`` and ``regressed`` name the cases that moved.
    """

    metric: Metric
    cases: int
    baseline: float
    candidate: float
    difference: float
    standard_error: float | None
    improved: tuple[str, ...]
    regressed: tuple[str, ...]


class Comparison(_Record):
    """The per-metric differences between a baseline run and a candidate run."""

    baseline_model: str
    candidate_model: str
    deltas: tuple[MetricDelta, ...]


def compare_runs(baseline: RunReport, candidate: RunReport) -> Comparison:
    """Compare two runs case by case.

    Raises:
        IncomparableRunsError: If the runs used different datasets.
    """
    if baseline.dataset_digest != candidate.dataset_digest:
        message = "the runs used different datasets and cannot be compared"
        raise IncomparableRunsError(message)
    before = {case.case_id: case.mean_scores() for case in baseline.cases}
    after = {case.case_id: case.mean_scores() for case in candidate.cases}
    deltas = []
    for metric in Metric:
        pairs = {
            case_id: (before[case_id][metric], after[case_id][metric])
            for case_id in before
            if metric in before[case_id] and metric in after[case_id]
        }
        if pairs:
            deltas.append(_delta(metric, pairs))
    return Comparison(
        baseline_model=baseline.model, candidate_model=candidate.model, deltas=tuple(deltas)
    )


def _delta(metric: Metric, pairs: dict[str, tuple[float, float]]) -> MetricDelta:
    differences = {case_id: after - before for case_id, (before, after) in pairs.items()}
    return MetricDelta(
        metric=metric,
        cases=len(pairs),
        baseline=statistics.fmean(before for before, _ in pairs.values()),
        candidate=statistics.fmean(after for _, after in pairs.values()),
        difference=statistics.fmean(differences.values()),
        standard_error=standard_error(list(differences.values())),
        improved=tuple(case_id for case_id, change in differences.items() if change > 0),
        regressed=tuple(case_id for case_id, change in differences.items() if change < 0),
    )


def render_comparison(comparison: Comparison) -> str:
    """Format a comparison as a plain-text table followed by the cases that moved."""
    lines = [
        f"baseline {comparison.baseline_model}  candidate {comparison.candidate_model}",
        f"{'metric':<26}{'cases':>6}{'base':>8}{'cand':>8}{'diff':>8}{'se':>8}",
    ]
    lines.extend(
        f"{delta.metric.value:<26}{delta.cases:>6}{delta.baseline:>8.2f}{delta.candidate:>8.2f}"
        f"{delta.difference:>+8.2f}{format_error(delta.standard_error):>8}"
        for delta in comparison.deltas
    )
    for delta in comparison.deltas:
        lines.extend(f"improved  {delta.metric.value}: {case_id}" for case_id in delta.improved)
        lines.extend(f"regressed {delta.metric.value}: {case_id}" for case_id in delta.regressed)
    return "\n".join(lines) + "\n"
