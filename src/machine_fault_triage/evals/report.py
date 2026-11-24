"""The report written by an evaluation run and its per-metric summary."""

import math
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.evals.scoring import Metric
from machine_fault_triage.models import TriageResult


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RepeatOutcome(_Record):
    """The result of one run of one case, with its scores."""

    result: TriageResult
    scores: dict[Metric, float]


class CaseOutcome(_Record):
    """Every repeat of one case."""

    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    repeats: tuple[RepeatOutcome, ...] = Field(min_length=1)

    def mean_scores(self) -> dict[Metric, float]:
        """Average each metric over the repeats in which it applied."""
        return {
            metric: statistics.fmean(values)
            for metric in Metric
            if (
                values := [
                    repeat.scores[metric] for repeat in self.repeats if metric in repeat.scores
                ]
            )
        }

    def passed_every_repeat(self, metric: Metric) -> bool:
        """Tell whether the metric scored 1 in every repeat."""
        return all(repeat.scores.get(metric) == 1 for repeat in self.repeats)


class MetricSummary(_Record):
    """One metric aggregated over the cases it applied to.

    ``mean`` averages the per-case means, and ``standard_error`` describes the
    spread between cases. ``passed_every_repeat`` counts the cases that scored
    1 in all repeats, which is what a user who runs the triage once relies on.
    """

    metric: Metric
    cases: int = Field(ge=1)
    mean: float
    standard_error: float | None
    passed_every_repeat: int = Field(ge=0)


class RunReport(_Record):
    """Everything needed to inspect one evaluation run or compare it with another."""

    schema_version: Literal[1] = 1
    dataset: str
    dataset_digest: str
    model: str
    library_version: str
    repeats: int = Field(ge=1)
    cases: tuple[CaseOutcome, ...]


def summarize(report: RunReport) -> tuple[MetricSummary, ...]:
    """Aggregate every metric that applied to at least one case."""
    summaries = []
    for metric in Metric:
        outcomes = [case for case in report.cases if metric in case.mean_scores()]
        if not outcomes:
            continue
        means = [case.mean_scores()[metric] for case in outcomes]
        summaries.append(
            MetricSummary(
                metric=metric,
                cases=len(means),
                mean=statistics.fmean(means),
                standard_error=standard_error(means),
                passed_every_repeat=sum(case.passed_every_repeat(metric) for case in outcomes),
            )
        )
    return tuple(summaries)


def standard_error(values: list[float]) -> float | None:
    """Return the standard error of the mean, or nothing for fewer than two values."""
    if len(values) < 2:  # noqa: PLR2004
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def render_summary(report: RunReport) -> str:
    """Format the summary of a run as a plain-text table."""
    lines = [
        f"dataset {report.dataset}  model {report.model}",
        f"{len(report.cases)} cases, {report.repeats} repeats",
        f"{'metric':<26}{'cases':>6}{'mean':>8}{'se':>8}{'every repeat':>14}",
    ]
    lines.extend(
        f"{summary.metric.value:<26}{summary.cases:>6}{summary.mean:>8.2f}"
        f"{format_error(summary.standard_error):>8}"
        f"{f'{summary.passed_every_repeat}/{summary.cases}':>14}"
        for summary in summarize(report)
    )
    return "\n".join(lines) + "\n"


def format_error(value: float | None) -> str:
    """Format a standard error for a table cell."""
    return "-" if value is None else f"{value:.2f}"
