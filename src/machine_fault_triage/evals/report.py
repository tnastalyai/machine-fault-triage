"""The report written by an evaluation run and its per-metric summary."""

import math
import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.evals.scoring import Metric
from machine_fault_triage.models import TriageResult, Usage


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RepeatOutcome(_Record):
    """The result of one run of one case, with its scores."""

    result: TriageResult
    scores: dict[Metric, float]
    seconds: float = Field(ge=0)
    usage: Usage | None = None


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


class Pricing(_Record):
    """Prices in US dollars per million tokens.

    ``cached_input`` is the price of input tokens read from the provider's
    prompt cache; without it they are priced as ordinary input. ``embedding``
    is the price of the embedding model; without it embedding tokens are
    left out of the cost.
    """

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cached_input: float | None = Field(default=None, ge=0)
    embedding: float | None = Field(default=None, ge=0)


class RunReport(_Record):
    """Everything needed to inspect one evaluation run or compare it with another."""

    schema_version: Literal[1] = 1
    dataset: str
    dataset_digest: str
    model: str
    library_version: str
    prompt_version: int | None = None
    prompt_digest: str | None = None
    page_selection: str | None = None
    repeats: int = Field(ge=1)
    pricing: Pricing | None = None
    cases: tuple[CaseOutcome, ...]

    def runs(self) -> tuple[RepeatOutcome, ...]:
        """Return every repeat of every case."""
        return tuple(repeat for case in self.cases for repeat in case.repeats)

    def usage(self) -> Usage | None:
        """Total the tokens of the runs that reported usage, if any did."""
        reported = [run.usage for run in self.runs() if run.usage is not None]
        if not reported:
            return None
        return Usage(
            input_tokens=sum(usage.input_tokens for usage in reported),
            output_tokens=sum(usage.output_tokens for usage in reported),
            cached_input_tokens=sum(usage.cached_input_tokens for usage in reported),
        )

    def embedding_tokens(self) -> int:
        """Total the tokens that page selection spent on embeddings."""
        return sum(run.result.embedding_tokens for run in self.runs())

    def cost(self) -> float | None:
        """Price the reported tokens in US dollars, when pricing and usage are known."""
        usage = self.usage()
        if usage is None or self.pricing is None:
            return None
        prices = self.pricing
        cached_price = prices.input if prices.cached_input is None else prices.cached_input
        return (
            (usage.input_tokens - usage.cached_input_tokens) * prices.input
            + usage.cached_input_tokens * cached_price
            + usage.output_tokens * prices.output
            + self.embedding_tokens() * (prices.embedding or 0)
        ) / 1_000_000


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


def _prompt_label(report: RunReport) -> str:
    if report.prompt_version is None:
        return "unknown"
    return f"v{report.prompt_version} ({report.prompt_digest})"


def _pages_label(report: RunReport) -> str:
    return f"pages: {report.page_selection or 'all'}"


def render_summary(report: RunReport) -> str:
    """Format the summary of a run as a plain-text table."""
    lines = [
        f"dataset {report.dataset}  model {report.model}  prompt {_prompt_label(report)}",
        f"{len(report.cases)} cases, {report.repeats} repeats, {_pages_label(report)}",
        f"{'metric':<26}{'cases':>6}{'mean':>8}{'se':>8}{'every repeat':>14}",
    ]
    lines.extend(
        f"{summary.metric.value:<26}{summary.cases:>6}{summary.mean:>8.2f}"
        f"{format_error(summary.standard_error):>8}"
        f"{f'{summary.passed_every_repeat}/{summary.cases}':>14}"
        for summary in summarize(report)
    )
    lines.append(_resources(report))
    return "\n".join(lines) + "\n"


def _resources(report: RunReport) -> str:
    runs = report.runs()
    parts = [f"median {statistics.median(run.seconds for run in runs):.1f} s per run"]
    usage = report.usage()
    if usage is not None:
        parts.append(f"{usage.input_tokens} input and {usage.output_tokens} output tokens")
        if usage.cached_input_tokens:
            parts.append(f"{usage.cached_input_tokens} of the input read from cache")
    if report.embedding_tokens():
        priced = report.pricing is not None and report.pricing.embedding is not None
        left_out = "" if priced or report.cost() is None else ", not in the cost"
        parts.append(f"{report.embedding_tokens()} embedding tokens{left_out}")
    cost = report.cost()
    if cost is not None:
        parts.append(f"${cost:.2f} in total, ${cost / len(runs):.4f} per run")
    return ", ".join(parts)


def format_error(value: float | None) -> str:
    """Format a standard error for a table cell."""
    return "-" if value is None else f"{value:.2f}"
