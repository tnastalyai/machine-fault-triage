# machine-fault-triage

Evidence-grounded fault triage for field service: heat pumps, HVAC, agricultural machinery.

Given a fault report (free text, optional photos of a display or nameplate), the device's manuals and its service history, the library returns a structured hint for the technician: probable causes, checks to run, the likely fix and parts to bring. Every cause points to a verified quote from a manual page or to a past service record. When the evidence is missing, the result says so instead of guessing.

> **Status: pre-alpha.** The result models are in place. Ingestion, retrieval, diagnosis and the evaluation harness are being built in the order listed under [Roadmap](#roadmap). Nothing here is ready for production use.

## Why

A technician usually learns what is wrong only after arriving on site. The information needed for a first hypothesis often already exists: the error code is on a photo, its meaning is on page 77 of a manual, and the same model failed the same way last spring. This library connects those three sources before the visit.

The hard part is not generating a plausible answer. It is knowing when the answer is supported, and saying "not enough data" when it is not. The design is built around that constraint.

## Design principles

- **Grounded or flagged.** A cause is either backed by a verified source or explicitly marked as general knowledge with low confidence. The model type enforces this.
- **Abstention is a valid result.** `not_enough_data` with questions for the customer is a first-class outcome, not an error.
- **Untrusted input stays data.** Fault reports and document text never become instructions. Tools are read-only.
- **Measured, not assumed.** Every change to prompts, models or retrieval is judged against an evaluation set with reported uncertainty.
- **Bring your own storage and model.** Documents, vector store, service history and the model provider are interfaces.

## Result shape

```python
from machine_fault_triage import (
    Cause,
    Confidence,
    DocumentSource,
    EvidenceLevel,
    TriageResult,
    TriageStatus,
)

result = TriageResult(
    status=TriageStatus.OK,
    summary="Error E7 points to the flow sensor.",
    evidence_level=EvidenceLevel.MANUAL,
    language="en",
    causes=(
        Cause(
            title="Flow sensor fault",
            confidence=Confidence.HIGH,
            why="The service manual maps E7 to the flow sensor circuit.",
            checks=("Measure sensor resistance at connector X4.",),
            fix="Replace the flow sensor.",
            sources=(
                DocumentSource(
                    document_id="service-manual",
                    document_title="AquaHeat 12 service manual",
                    page=77,
                    section="7.3 Error codes",
                    quote="E7: flow sensor open or short circuit",
                    verified=True,
                ),
            ),
        ),
    ),
    parts_to_bring=("Flow sensor",),
)
```

Each source carries the evidence a technician can check: the document title, the page, the section and the exact quote. `verified` is set only after the quote has been found on that page.

A cause without a verified source cannot claim more than low confidence; constructing one raises a validation error.

## Roadmap

1. Result models and project tooling
2. Evaluation harness and a hand-written seed dataset
3. Baseline: whole documents in the prompt
4. Behaviour without documents: abstention and questions for the customer
5. Page-aware ingestion and hybrid retrieval
6. Quote verification and prompt-injection hardening
7. Device and model service history
8. Tool-using agent variant, compared against the fixed pipeline
9. Postgres adapter and full evaluation report

## Development

```bash
uv sync
uv run ruff format --check
uv run ruff check
uv run mypy
uv run pytest --cov
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions.

## License

[Apache-2.0](LICENSE)
